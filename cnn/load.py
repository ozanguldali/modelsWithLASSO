import torch

from cnn import device
from cnn.architect import ProCNN


def load_model(model, path):
    map_location = None if torch.cuda.is_available() else device
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()

    return model


if __name__ == '__main__':
    load_model(ProCNN(), "/cnn/ProCNN_dataset_kaggle_out.pth")
