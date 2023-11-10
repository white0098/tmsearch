import torch
import torch.nn as nn

import torchvision.models as models

class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name='VGG19'):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        
    def _build_model(self):
        if self.model_name == 'VGG19':
            # feature extractor
            model = models.vgg19(weights='DEFAULT').features.to(self.device).eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
        else:
            raise NotImplementedError
        return model

    def forward(self, x):
        return self.model(x)

    
    

if __name__ == '__main__':
    model = ImageFeatureExtractor(model_name='VGG19')