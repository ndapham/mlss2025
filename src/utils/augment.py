import random
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import imgaug as ia
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)

        self.aug = iaa.Sequential(
            iaa.SomeOf((1, 5), 
            [
            # blur

            sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
                                iaa.MotionBlur(k=3)])),
        
            # color
            sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
            sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
            sometimes(iaa.Invert(0.25, per_channel=0.5)),
            sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
            sometimes(iaa.Dropout2d(p=0.5)),
            sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
            sometimes(iaa.Add((-40, 40), per_channel=0.5)),

            sometimes(iaa.JpegCompression(compression=(5, 80))),
            
            # distort
            sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
            sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), 
    #                            rotate=(-5, 5), shear=(-5, 5), 
                                order=[0, 1], cval=(0, 255), 
                                mode=ia.ALL)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
            sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))]))
        ],

        random_order=True),
        random_order=True)
      
    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        img = Image.fromarray(img)
        return img
  

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
