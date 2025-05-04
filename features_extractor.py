import torch
from torchvision import models, transforms, datasets
from PIL import Image

class ResNet50Extractor:
    IMAGENET_MEAN = models.ResNet50_Weights.DEFAULT.transforms().mean 
    IMAGENET_STD = models.ResNet50_Weights.DEFAULT.transforms().std

    def __init__(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove classifier:
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) # Exclude fc
        #self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) # Exclude avgpool and fc 

        self.feature_extractor.eval()

    def extract(self, img_tensor: torch.Tensor) -> list[float]:
        features = None
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            features = features.view(features.size(0), -1) # Flatten
        return features[0]
    
    @staticmethod
    def get_image_tensor(img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
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
        img = ResNet50Extractor.get_image_tensor(img_path)
        features = extractor.extract(img)


