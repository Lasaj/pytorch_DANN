import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF
from torchvision import models
from torchvision.models.inception import Inception_V3_Weights
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.models.resnet import ResNet50_Weights


def get_iv3():
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.AuxLogits.fc = nn.Linear(768, 768)
    # model.fc = nn.Linear(2048, (3 * 28 * 28))
    model.fc = nn.Identity()
    return model


def get_densenet(use_xrv_weights=False):
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1, progress=True)
    model.classifier = nn.Identity()
    if use_xrv_weights:
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.load_state_dict(torch.load('./trained_models/torchxrayvision/densenet121-res224-all.pt'))

    return model


def get_resnet():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True)

    # reshape model to load dict
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 1)

    # fix state_dict keys to match model and load
    if torch.cuda.is_available():
        state_dict = torch.load('./trained_models/chest-x-ray-resnet50-model.pth')
    else:
        state_dict = torch.load('./trained_models/chest-x-ray-resnet50-model.pth', map_location=torch.device('cpu'))
    for key in list(state_dict.keys()):
        state_dict[key.replace('network.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)

    model.fc = nn.Identity()
    return model


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1, 3 * 28 * 28)
        return x


class Classifier(nn.Module):
    def __init__(self, in_features=3 * 28 * 28, out_features=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=out_features)
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x)


class Discriminator(nn.Module):
    def __init__(self, in_features=3 * 28 * 28):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return F.softmax(x)


def main():
    model = get_resnet()
    print(model)


if __name__ == '__main__':
    main()
