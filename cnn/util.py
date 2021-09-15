import sys

from torch import nn
from torchvision import models

from cnn import MODEL_NAME
from cnn.models import novelnet, proposednet

from util.logger_util import log


def prepare_proposednet(is_pre_trained, fine_tune, num_classes):
    model = proposednet.proposednet(pretrained=is_pre_trained, num_classes=num_classes)
    if fine_tune:
        frozen = nn.Sequential(
            model.features,
            model.avgpool
        )
        set_parameter_requires_grad(frozen)

    return model


def prepare_novelnet(is_pre_trained, fine_tune, num_classes):
    model = novelnet.novelnet(pretrained=is_pre_trained, num_classes=num_classes)
    if fine_tune:
        frozen = nn.Sequential(
            model.features
        )
        set_parameter_requires_grad(frozen)

    return model


def prepare_alexnet(is_pre_trained, fine_tune, num_classes):
    model = models.alexnet(pretrained=is_pre_trained,
                           num_classes=1000 if is_pre_trained else num_classes)
    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(3)]
        )
        set_parameter_requires_grad(frozen)

    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    custom_fc = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Linear(model.classifier[-1].out_features, num_classes)
    )

    model.classifier.add_module("custom_fc", custom_fc)

    return model


def prepare_resnet(model_name, is_pre_trained, fine_tune, num_classes):
    if model_name == models.resnet18.__name__:
        model = models.resnet18(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
    elif model_name == models.resnet34.__name__:
        model = models.resnet34(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)

    elif model_name == models.resnet50.__name__:
        model = models.resnet50(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
    elif model_name == models.resnet152.__name__:
        model = models.resnet152(pretrained=is_pre_trained,
                                 num_classes=1000 if is_pre_trained else num_classes)
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        set_parameter_requires_grad(frozen)

    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    custom_fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(model.fc.out_features, num_classes)
        )

    model.fc = nn.Sequential(
        model.fc,
        custom_fc
    )

    return model


def prepare_vgg(model_name, is_pre_trained, fine_tune, num_classes):
    if model_name == models.vgg16.__name__:
        model = models.vgg16_bn(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
        limit_frozen = 7
    elif model_name == models.vgg19.__name__:
        model = models.vgg19_bn(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
        limit_frozen = 7
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(limit_frozen)]
        )
        set_parameter_requires_grad(frozen)

    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    custom_fc = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Linear(model.classifier[-1].out_features, num_classes)
    )

    model.classifier.add_module("custom_fc", custom_fc)

    return model


def prepare_densenet(is_pre_trained, fine_tune, num_classes):
    model = models.densenet169(pretrained=is_pre_trained,
                               num_classes=1000 if is_pre_trained else num_classes)

    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(4)]
        )
        set_parameter_requires_grad(frozen)

    # model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    custom_fc = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Linear(model.classifier.out_features, num_classes)
    )

    model.classifier = nn.Sequential(
        model.classifier,
        custom_fc
    )

    return model


def prepare_googlenet(is_pre_trained, fine_tune, num_classes):
    model = models.googlenet(pretrained=is_pre_trained,
                             num_classes=1000 if is_pre_trained else num_classes)

    if fine_tune:
        frozen = nn.Sequential(
            model.conv1,
            model.maxpool1,
            model.conv2,
            model.conv3,
            model.maxpool2
        )
        set_parameter_requires_grad(frozen)

    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    custom_fc = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Linear(model.fc.out_features, num_classes)
    )

    model.fc = nn.Sequential(
        model.fc,
        custom_fc
    )

    return model


def prepare_squeezenet(model_name, is_pre_trained, fine_tune, num_classes):
    if model_name == models.squeezenet1_0.__name__:
        model = models.squeezenet1_0(pretrained=is_pre_trained,
                                     num_classes=1000 if is_pre_trained else num_classes)
    elif model_name == models.squeezenet1_1.__name__:
        model = models.squeezenet1_1(pretrained=is_pre_trained,
                                     num_classes=1000 if is_pre_trained else num_classes)
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            *[model.feeatures[i] for i in range(3)]
        )
        set_parameter_requires_grad(frozen)

    # model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=model.classifier[1].kernel_size)
    custom_fc = nn.Sequential(
        nn.Flatten(),
        nn.ReLU(inplace=True),
        nn.Dropout2d(),
        nn.Linear(1000, num_classes)
    )

    model.classifier.add_module("custom_fc", custom_fc)

    return model


def is_verified(acc):
    model_name = MODEL_NAME[0]

    if model_name == models.alexnet.__name__:
        verified = acc > 88.2  # 90.2

    elif model_name == models.resnet18.__name__:
        verified = acc > 90.2

    elif model_name == models.resnet34.__name__:
        verified = acc > 90.2

    elif model_name == models.resnet50.__name__:
        verified = acc > 88.2  # 92.16

    elif model_name == models.vgg16.__name__:
        verified = acc > 90.2

    elif model_name == models.vgg19.__name__:
        verified = acc > 90.2

    elif model_name == models.densenet169.__name__:
        verified = acc > 87.2

    else:
        verified = True

    return verified


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
