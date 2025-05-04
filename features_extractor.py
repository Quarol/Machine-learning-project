import torch
from torchvision import models, transforms, datasets
from PIL import Image
from typing import Union
import numpy as np

class ResNet50Extractor:
    IMAGENET_MEAN = models.ResNet50_Weights.DEFAULT.transforms().mean 
    IMAGENET_STD = models.ResNet50_Weights.DEFAULT.transforms().std

    def __init__(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove classifier:
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) # Exclude fc
        #self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) # Exclude avgpool and fc 

        self.feature_extractor.eval()

    # Use to extract features from image tensor
    def extract(self, img_tensor: torch.Tensor) -> list[float]:
        features = None
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            features = features.view(features.size(0), -1) # Flatten
        return features[0]
    
    # Use to create image tensor.
    @staticmethod
    def create_image_tensor(img_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(img_input, str):
            img = Image.open(img_input)
        elif isinstance(img_input, Image.Image):
            img = img_input
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:  # Grayscale
                img = Image.fromarray(img_input).convert('RGB')
            elif img_input.ndim == 3:
                if img_input.shape[2] == 3:
                    img = Image.fromarray(img_input.astype(np.uint8), 'RGB')
                elif img_input.shape[2] == 4:
                    img = Image.fromarray(img_input.astype(np.uint8), 'RGBA').convert('RGB')
                else:
                    raise ValueError("Unsupported channel number in numpy array.")
            else:
                raise ValueError("Unsupported numpy array shape.")
        else:
            raise TypeError("img_input must be a file path, PIL Image, or numpy array")

        if img.mode != 'RGB':
            img = img.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ResNet50Extractor.IMAGENET_MEAN,
                                 std=ResNet50Extractor.IMAGENET_STD)
        ])
        return transform(img).unsqueeze(0)

if __name__ == '__main__':
    image_paths = ['imgs/cat.jpg', 'imgs/breast.jpg', 'imgs/black.jpg']
    extractor = ResNet50Extractor()

    for img_path in image_paths:
        img = ResNet50Extractor.create_image_tensor(img_path)
        features = extractor.extract(img)
        print(len(features))
        print(features)
        print()
