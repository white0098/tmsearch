import cv2

import torch
import torch.nn as nn

import torchvision.models as models


class ImageFeatureExtractor(nn.Module):
    def __init__(self, model_name="VGG19"):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    def _preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # shape: (224, 224, 3)
        img = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        )  # shape: (1, 3, 224, 224)
        return img

    def _build_model(self):
        if self.model_name == "VGG19":
            # feature extractor
            model = models.vgg19(weights="DEFAULT").features.to(self.device).eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        elif self.model_name == "VIT_B_16":
            # feature extractor
            model = models.vit_b_16(weights="DEFAULT").features.to(self.device).eval()
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        else:
            raise NotImplementedError
        return model

    def calculate_similarity(self, x1, x2):
        # f1, f2: shape: (1, C, H, W)
        # calculate cosine similarity
        f1 = self.model(x1)
        f2 = self.model(x2)
        similarity = torch.cosine_similarity(f1, f2, dim=0)
        # convert to 0 ~ 1
        similarity = (similarity + 1) / 2
        return similarity


    def forward(self, x):
        preprocess_x = self._preprocess(x)
        return self.model(preprocess_x)


if __name__ == "__main__":
    model = ImageFeatureExtractor(model_name="VGG19")
