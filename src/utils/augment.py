import random
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
  

class CustomAugmentation:
    def __init__(self, p=0.3):
        self.p = p
        self.convert_to_tensor = transforms.PILToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image, scribble, ground_truth):
        image, scribble, ground_truth = self.ResizeRandomCrop(image, scribble, ground_truth, (0.8, 0.8), self.p)
        image, scribble, ground_truth = self.HorizontalFlip(image, scribble, ground_truth, self.p)
        image, scribble, ground_truth = self.VerticalFlip(image, scribble, ground_truth, self.p)
        image, scribble, ground_truth = self.ColorJitter(image, scribble, ground_truth, self.p)
        image, scribble, ground_truth = self.RandomGrayscale(image, scribble, ground_truth, self.p)
        image, scribble, ground_truth = self.GaussianBlur(image, scribble, ground_truth, self.p)

        return image, scribble, ground_truth

    def ResizeRandomCrop(self, image, scribble, ground_truth, crop_scale,  p=0.3):
        if random.random() < p:
            h_scale, w_scale = crop_scale
            w, h = image.size
            resize = transforms.Resize(size=(h, w))
            # Crop 
            crop_size = (round(h * h_scale), round(w * w_scale))
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=crop_size)
            image = TF.crop(image, i, j, h, w)
            scribble = TF.crop(scribble, i, j, h, w)
            ground_truth =  TF.crop(ground_truth, i, j, h, w)
            # Resize 
            image = resize(image)
            scribble = resize(scribble)
            ground_truth = resize(ground_truth)

        return image, scribble, ground_truth   

    def HorizontalFlip(self, image, scribble, ground_truth, p=0.3):
        if random.random() < p:
            image = TF.hflip(image)
            scribble = TF.hflip(scribble)
            ground_truth = TF.hflip(ground_truth)

        return image, scribble, ground_truth  


    def VerticalFlip(self, image, scribble, ground_truth, p=0.3):
        if random.random() < p:
            image = TF.vflip(image)
            scribble = TF.vflip(scribble)
            ground_truth = TF.vflip(ground_truth)

        return image, scribble, ground_truth  
     
    def RandomGrayscale(self, image, scribble, ground_truth, p=0.3):
        convert_to_grayscale = transforms.RandomGrayscale(p=p)
        image = convert_to_grayscale(image)
        return image, scribble, ground_truth  
    
    def GaussianBlur(self, image, scribble, ground_truth, p=0.3):
        if random.random() < p:
            gauss_blur = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            image = gauss_blur(image)
        return image, scribble, ground_truth 
    
    def ColorJitter(self, image, scribble, ground_truth, p=0.3):
        if random.random() < p:
            jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            image = jitter(image)
        return image, scribble, ground_truth
    
    def GlobalAugment(self,image, scribble, ground_truth, crop_scale=(0.4, 1.0)):
        image, scribble, ground_truth = self.ResizeRandomCrop(image, scribble, ground_truth, crop_scale, 1)
        image, scribble, ground_truth = self.HorizontalFlip(image, scribble, ground_truth, 0.5)
        image, scribble, ground_truth = self.VerticalFlip(image, scribble, ground_truth, 0)
        image, scribble, ground_truth = self.ColorJitter(image, scribble, ground_truth, 1)
        image, scribble, ground_truth = self.RandomGrayscale(image, scribble, ground_truth,0.2)
        image, scribble, ground_truth = self.GaussianBlur(image, scribble, ground_truth, 1)
        image = self.transform(image)
        scribble = self.convert_to_tensor(scribble)
        ground_truth = self.convert_to_tensor(ground_truth)
        return image, scribble, ground_truth
    
    def LocalAugment(self,image, scribble, ground_truth, crop_scale=(0.05, 0.4)):
        image, scribble, ground_truth = self.ResizeRandomCrop(image, scribble, ground_truth, crop_scale, 1)
        image, scribble, ground_truth = self.HorizontalFlip(image, scribble, ground_truth, 0.5)
        image, scribble, ground_truth = self.VerticalFlip(image, scribble, ground_truth, 0)
        image, scribble, ground_truth = self.ColorJitter(image, scribble, ground_truth, 1)
        image, scribble, ground_truth = self.RandomGrayscale(image, scribble, ground_truth,0.2)
        image, scribble, ground_truth = self.GaussianBlur(image, scribble, ground_truth, 1)
        image = self.transform(image)
        scribble = self.convert_to_tensor(scribble)
        ground_truth = self.convert_to_tensor(ground_truth)
        return image, scribble, ground_truth
