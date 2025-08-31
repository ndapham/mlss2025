import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.utils.augment import CustomAugmentation


class ScribbleDataset(Dataset):
    def __init__(self, images, scribbles, fnames, annotations=None, augment_rate=0.3):
        
        self.images = images
        self.scribbles = scribbles
        self.annotations = annotations
        self.fnames = fnames
        self.augment_rate = augment_rate
        self.augmentation = CustomAugmentation(p=0.3)
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        scribble = Image.fromarray(self.scribbles[idx])
        PILToTensor = transforms.PILToTensor()
        fname = self.fnames[idx]

        if self.annotations is None:
            image = self.transform(image)
            image = torch.as_tensor(image, dtype=torch.float32)
            scribble = torch.as_tensor(PILToTensor(scribble), dtype=torch.float32)
            return image, scribble, fname
        
        annotation = Image.fromarray(self.annotations[idx])
        if self.augment_rate:
            image, scribble, annotation = self.augmentation(image, scribble, annotation)
        image = self.transform(image)
        image = torch.as_tensor(image, dtype=torch.float32)
        scribble = torch.as_tensor(PILToTensor(scribble), dtype=torch.float32)
        annotation = torch.as_tensor(PILToTensor(annotation), dtype=torch.float32)
        return image, scribble, annotation, fname

    def __len__(self):
        return len(self.images)
