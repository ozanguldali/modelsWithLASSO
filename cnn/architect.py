import torch
import torch.nn as nn

from cnn import ROOT_DIR, device

__all__ = ['ProCNN', 'procnn']


class ProCNN(nn.Module):
    def __init__(self):
        super(ProCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, stride=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=128),
            nn.AdaptiveMaxPool2d(output_size=37),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),
            nn.AdaptiveMaxPool2d(output_size=10),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),
            nn.AdaptiveMaxPool2d(output_size=3),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),
            nn.AdaptiveMaxPool2d(output_size=1),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=512),
            nn.AdaptiveMaxPool2d(output_size=1),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(1 * 1 * 512, 1024)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(1 * 1 * 1024, 1024)
        )

        self.fc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(1 * 1 * 1024, 2)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(train * train * 512, 1024),
        #
        #     nn.Dropout2d(),
        #     nn.Linear(train * train * 1024, 1024),
        #
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(),
        #     nn.Linear(train * train * 1024, 3)
        # )

        self.softMax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softMax(x)

        return x


def procnn(pretrained=False, dataset_folder="dataset", **kwargs):
    r"""ProCNN model architecture from the

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param dataset_folder: pth file name
    """
    model = ProCNN()

    if pretrained:
        map_location = None if torch.cuda.is_available() else device
        model.load_state_dict(torch.load(ROOT_DIR+"/ProCNN_"+dataset_folder+"_out.pth", map_location=map_location))
        # model.load_state_dict(torch.load(ROOT_DIR+"/ProCNN_"+dataset_folder+"_out.pth"))
        # model.eval()

    return model
