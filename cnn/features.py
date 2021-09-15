import numpy as np

from torch import nn
from tqdm import tqdm

from cnn import device
from util.logger_util import log


def extract_features(data_loader, feature_extractor):
    X = []

    for images, labels in tqdm(data_loader):
        images = images.to(device)

        if feature_extractor is not None:
            X.extend(
                [features.tolist() for features in feature_extractor(images)]
            )

    return X


# Deep Features from FC2
def proposednet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        model.flatten,
        model.fc1,
        model.fc2
    )

    return feature_extractor


# Deep Features from FC2
def novelnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.flatten,
        model.fc1,
        model.fc2
    )

    return feature_extractor


# Deep Features from FC2
def alexnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *[model.classifier[i] for i in range(7)]
    )

    return feature_extractor


# Deep Features from Convolution Base
def resnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        nn.Flatten(),
        *[model.fc[i] for i in range(1)]
    )

    return feature_extractor


# Deep Features from FC2
def vgg_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *[model.classifier[i] for i in range(7)]
    )

    return feature_extractor


# Deep Features from Convolution Base
def densenet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        *[model.classifier[i] for i in range(1)]
    )

    return feature_extractor


# Deep Features from Convolution Base
def googlenet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.conv1,
        model.maxpool1,
        model.conv2,
        model.conv3,
        model.maxpool2,
        model.inception3a,
        model.inception3b,
        model.maxpool3,
        model.inception4a,
        # model.aux1,
        model.inception4b,
        model.inception4c,
        model.inception4d,
        # model.aux2,
        model.inception4e,
        model.maxpool4,
        model.inception5a,
        model.inception5b,
        model.avgpool,
        nn.Flatten(),
        model.dropout
    )

    return feature_extractor


def squeezenet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.classifier,
        nn.Flatten()
    )

    return feature_extractor


def feature_clean_withinclass(X, y, class0_size, class1_size, class0):
    np_x = np.reshape(X, (len(X), len(X[0])))

    nr, nc = np_x.shape

    zero_std = []
    for c in range(nc):
        column_i_class0 = np.zeros((1, class0_size))
        column_i_class1 = np.zeros((1, class1_size))
        index_i_0 = 0
        index_i_1 = 0
        for r in range(nr):
            if y[r] == class0:
                column_i_class0[0, index_i_0] = np_x[r, c]
                index_i_0 += 1
            else:
                column_i_class1[0, index_i_1] = np_x[r, c]
                index_i_1 += 1

        if np.std(column_i_class0) == 0.0 or np.std(column_i_class1) == 0.0:
            zero_std.append(c)

    len_zero_std = len(zero_std)
    log.info("Number of features having within-class standard deviation as 0: " + str(len_zero_std))
    if len_zero_std != 0:
        log.info("Eliminating 0 within-class standard deviation feature columns")
        np_x = np.delete(np_x, zero_std, axis=1)
        del X
        X = np.ndarray.tolist(np_x)

    return X


def feature_clean(X):
    np_x = np.reshape(X, (len(X), len(X[0])))

    nr, nc = np_x.shape

    zero_std = []
    for c in range(nc):
        column_i = np.zeros((1, nc))
        index = 0
        for r in range(nr):
            column_i[0, index] = np_x[r, c]

        if np.std(column_i) == 0.0:
            zero_std.append(c)

    len_zero_std = len(zero_std)
    log.info("Number of features having within-class standard deviation as 0: " + str(len_zero_std))
    if len_zero_std != 0:
        log.info("Eliminating 0 within-class standard deviation feature columns")
        np_x = np.delete(np_x, zero_std, axis=1)
        del X
        X = np.ndarray.tolist(np_x)

    return X
