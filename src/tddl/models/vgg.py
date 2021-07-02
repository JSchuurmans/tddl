import torch.nn as nn
from torchvision import models

# VGG16 based network for classifying between dogs and cats.
# After training this will be an over parameterized network,
# with potential to shrink it.
class ModifiedVGG16Model(nn.Module):
    def __init__(self, model=None):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x