from torchvision.models import resnet18
from torch.nn import Conv2d, Linear

def get_resnet_from_torch(in_channels, num_classes):

    model = resnet18()

    model.conv1 = Conv2d(
        in_channels=in_channels, 
        out_channels=64,
        kernel_size=(3,3),
        stride=(1,1),
        padding=(1,1),
        bias=False,
    )

    model.fc = Linear(
        in_features=512,
        out_features=num_classes,
        bias=True,
    )

    return model