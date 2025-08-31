import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Callable

from src.utils.augment import CustomAugmentation


class DINO(nn.Module):
    def __init__(self, student_arch: Callable, teacher_arch: Callable, device: torch.device):
        """
        Args:
            student_arch (nn.Module): Network for student_arch
            teacher_arch (nn.Module): Network for teacher_arch
            device: torch.device ('cuda' or 'cpu')
        """
        super(DINO, self).__init__()
    
        self.student = student_arch().to(device)
        self.teacher = teacher_arch().to(device)
        self.teacher.load_state_dict(self.student.state_dict())

        # Initialize center as buffer to avoid backpropagation
        output_dim = 375 * 500
        # self.register_buffer('center', torch.zeros(1, student_arch().output_dim))
        self.register_buffer('center', torch.zeros(1, output_dim))

        # Ensure the teacher parameters do not get updated during backprop
        for param in self.teacher.parameters():
            param.requires_grad = False

    @staticmethod
    def distillation_loss(student_output, teacher_output, center, tau_s, tau_t):
        """
        Calculates distillation loss with centering and sharpening (function H in pseudocode).
        """
        # Detach teacher output to stop gradients.
        batch_size = teacher_output.shape[0]
        teacher_output = teacher_output.detach().view(batch_size, -1)

        # Center and sharpen teacher's outputs
        teacher_probs = F.softmax((teacher_output - center) / tau_t, dim=1)

        # Sharpen student's outputs
        student_output = student_output.view(batch_size, -1)
        student_probs = F.log_softmax(student_output / tau_s, dim=1)

        # Calculate cross-entropy loss between student's and teacher's probabilities.
        loss = - (teacher_probs * student_probs).sum(dim=1).mean()
        return loss
    

    def teacher_update(self, beta: float):
        for teacher_params, student_params in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_params.data.mul_(beta).add_(student_params.data, alpha=(1 - beta))


def global_augment(images, scribbles, annotations, num_crops=2):
    convert_to_PIL = transforms.ToPILImage()

    augment = CustomAugmentation()
    augment_list = []
    for _ in range(num_crops):
        for i in range(images.shape[0]):
            image, scribble, annotation = images[i], scribbles[i], annotations[i]
            image = convert_to_PIL(image)
            scribble = convert_to_PIL(scribble)
            annotation = convert_to_PIL(annotation)
            aug_image, aug_scribble, aug_annotation = augment.GlobalAugment(image, scribble, annotation)
            augment_input = torch.cat([aug_image, aug_scribble], dim=0)
            augment_list.append(augment_input)

    return torch.stack(augment_list, dim=0)

def multiple_local_augments(images, scribbles, annotations, num_crops=2):
    convert_to_PIL = transforms.ToPILImage()

    augment = CustomAugmentation()
    augment_list = []
    for _ in range(num_crops):
        for i in range(images.shape[0]):
            image, scribble, annotation = images[i], scribbles[i], annotations[i]
            image = convert_to_PIL(image)
            scribble = convert_to_PIL(scribble)
            annotation = convert_to_PIL(annotation)
            aug_image, aug_scribble, aug_annotation = augment.GlobalAugment(image, scribble, annotation)
            augment_input = torch.cat([aug_image, aug_scribble], dim=0)
            augment_list.append(augment_input)
    return torch.stack(augment_list, dim=0)

def train_dino(dino: DINO,
               data_loader: DataLoader,
               optimizer: Optimizer,
               device: torch.device,
               num_epochs,
               n_global_crops=2,
               n_local_crops=2,
               tps=0.9,
               tpt= 0.04,
               beta= 0.9,
               m= 0.9,
               ):
        """
        Args:
        dino: DINO Module
        data_loader (nn.Module): Dataloader for training
        optimizer (nn.optimizer): Optimizer for optimization (SGD etc.)
        defice (torch.device): 'cuda', 'cpu'
        num_epochs: Number of Epochs
        tps (float): tau for sharpening student logits
        tpt: for sharpening teacher logits
        beta (float): moving average decay 
        m (float): center moveing average decay
        """
    
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}/{num_epochs}")
            for images, scribbles, annos, fnames in data_loader:
                B, C, W, H = images.shape
                global_crops = global_augment(images, scribbles, annos, n_global_crops)
                local_crops = multiple_local_augments(images, scribbles, annos, n_local_crops)

                student_output = dino.student(local_crops).to(device)

                with torch.no_grad():
                    teacher_output = dino.teacher(global_crops.to(device))
                
                # Calculating loss
                total_loss=0
                n_loss=0
                
                for ti, t_out in enumerate(teacher_output.chunk(n_global_crops)):
                    for si, s_out in enumerate(student_output.chunk(n_local_crops)):
                        if ti != si: # Skipping the outputs where student crops index matches with teachers, code reference https://github.com/facebookresearch/dino/blob/main/main_dino.py#L396
                            total_loss += dino.distillation_loss(s_out, t_out, dino.center, tps, tpt)
                            n_loss += 1
                total_loss /= n_loss  # n_loss = total number - 2 for the two times loss was skipped above

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Update the teacher network parameters
                dino.teacher_update(beta)
                
                # Update the center
                with torch.no_grad():
                    dino.center = m * dino.center + (1 - m) * teacher_output.view(teacher_output.shape[0], -1).mean(dim=0)
        return dino.student
