from cnn import ROOT_DIR
from cnn.dataset import set_dataset, set_loader
from cnn.features import *
from cnn.models import proposednet, novelnet
from cnn.util import *

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from util.logger_util import log


def set_dataset_and_loaders(dataset_folder, augmented, batch_size, img_size, num_workers, shuffle_train=True, normalize=None):

    dataset_dir = ROOT_DIR.split("cnn")[0]

    log.info("Setting train data")
    train_data = set_dataset(folder=dataset_dir + dataset_folder + '/train', size=img_size, augmented=augmented, normalize=normalize)
    log.info("Train data length: %d" % len(train_data))
    log.info("Setting test data")
    test_data = set_dataset(folder=dataset_dir + dataset_folder + '/test', size=img_size, normalize=normalize)
    log.info("Test data length: %d" % len(test_data))

    log.info("Setting train loader")
    train_loader = set_loader(dataset=train_data, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    log.info("Setting test loader")
    test_loader = set_loader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, train_loader, test_data, test_loader


def get_model(model_name, is_pre_trained, fine_tune, num_classes):
    log.info("Instantiate the model")

    if model_name == novelnet.novelnet.__name__:
        model = prepare_novelnet(is_pre_trained, fine_tune, num_classes)

    elif model_name == proposednet.proposednet.__name__:
        model = prepare_proposednet(is_pre_trained, fine_tune, num_classes)

    elif model_name == models.alexnet.__name__:
        model = prepare_alexnet(is_pre_trained, fine_tune, num_classes)

    elif model_name in (models.resnet18.__name__, models.resnet34.__name__, models.resnet50.__name__, models.resnet152.__name__):
        model = prepare_resnet(model_name, is_pre_trained, fine_tune, num_classes)

    elif model_name in (models.vgg16.__name__, models.vgg19.__name__):
        model = prepare_vgg(model_name, is_pre_trained, fine_tune, num_classes)

    elif model_name == models.densenet169.__name__:
        model = prepare_densenet(is_pre_trained, fine_tune, num_classes)

    elif model_name == models.googlenet.__name__:
        model = prepare_googlenet(is_pre_trained, fine_tune, num_classes)

    elif model_name in (models.squeezenet1_0.__name__, models.squeezenet1_1.__name__):
        model = prepare_squeezenet(model_name, is_pre_trained, fine_tune, num_classes)

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    return model


def get_feature_extractor(model_name, model):

    if model_name == models.alexnet.__name__:
        feature_extractor = alexnet_feature_extractor(model)

    elif model_name in (models.resnet18.__name__, models.resnet34.__name__, models.resnet50.__name__, models.resnet152.__name__):
        feature_extractor = resnet_feature_extractor(model)

    elif model_name in (models.vgg16.__name__, models.vgg19.__name__):
        feature_extractor = vgg_feature_extractor(model)

    elif model_name == models.densenet169.__name__:
        feature_extractor = densenet_feature_extractor(model)

    elif model_name == models.googlenet.__name__:
        feature_extractor = googlenet_feature_extractor(model)

    elif model_name in (models.squeezenet1_0.__name__, models.squeezenet1_1.__name__):
        feature_extractor = squeezenet_feature_extractor(model)

    elif model_name == proposednet.proposednet.__name__:
        feature_extractor = proposednet_feature_extractor(model)

    elif model_name == novelnet.novelnet.__name__:
        feature_extractor = novelnet_feature_extractor(model)

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    return feature_extractor


def get_grad_update_params(model, feature_extract):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    return params_to_update


def get_metric_results(label_list, prediction_list):

    conf_matrix = confusion_matrix(label_list, prediction_list)
    log.info("Confusion Matrix:\n" + str(conf_matrix))

    auc_score = roc_auc_score(label_list, prediction_list)
    log.info("AUC Score:\n" + str(auc_score))

    report = classification_report(label_list, prediction_list)
    log.info("Classification Report:\n" + str(report))

    return conf_matrix, auc_score, report
