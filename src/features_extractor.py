import torch
from torchvision import models, transforms, datasets
from PIL import Image
from typing import Union
import numpy as np

class ResNet50Extractor:
    WEIGHTS = models.ResNet50_Weights.DEFAULT
    IMAGENET_MEAN = WEIGHTS.transforms().mean 
    IMAGENET_STD = WEIGHTS.transforms().std 

    def __init__(self, exclude_avgpool: bool = False):
        model = models.resnet50(weights=ResNet50Extractor.WEIGHTS)
        
        # Remove classifier:
        if exclude_avgpool:
            # Exclude avgpool and fc 
            self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) 
        else:
            # Exclude fc
            self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) 

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

def get_resnet50_features(input_images: list[Union[str, Image.Image, np.ndarray]]) -> list[list[float]]:
    extractor = ResNet50Extractor()
    features_lists = []

    for img in input_images:
        img_tensor = ResNet50Extractor.create_image_tensor(img)
        features_tensor = extractor.extract(img_tensor)
        features = features_tensor[0].tolist()
        features_lists.append(features)
    
    return features_lists


