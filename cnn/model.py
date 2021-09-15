import sys

import torch.utils.data.dataset
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from cnn import device, ROOT_DIR, SAVE_FILE, MODEL_NAME, saver_semaphore
from cnn.helper import get_model, get_grad_update_params
from cnn.load import load_model
from cnn.models import proposednet, novelnet
from cnn.save import save_model
from cnn.summary import get_summary
from cnn.test import test_model
from cnn.train import train_model
from cnn.util import is_verified
from util.file_util import path_exists

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, optimizer_name, is_pre_trained, fine_tune, num_epochs, train_loader, test_loader,
              validation_freq, lr, momentum, weight_decay, update_lr=False, save=False):
    collect_garbage()

    MODEL_NAME[0] = model_name

    if isinstance(train_loader.dataset, torch.utils.data.dataset.ConcatDataset):
        num_classes = len(train_loader.dataset.datasets[1].classes)
    else:
        num_classes = len(train_loader.dataset.classes)

    # instantiate the model
    model = get_model(model_name=model_name, is_pre_trained=is_pre_trained, fine_tune=fine_tune,
                      num_classes=num_classes)

    log.info("Setting the model to device")
    model = model.to(device)

    log.info("The summary:")
    if "densenet" not in model_name:
        get_summary(model, train_loader)
    else:
        log.info(model)

    collect_garbage()

    log.info("Setting the metric")
    metric = nn.CrossEntropyLoss()

    model_parameters = get_grad_update_params(model, fine_tune)

    if optimizer_name == optim.Adam.__name__:
        optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == optim.AdamW.__name__:
        optimizer = optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == optim.SGD.__name__:
        optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum)
    else:
        log.fatal("not implemented optimizer name: {}".format(optimizer_name))
        sys.exit(1)

    log.info("Setting the optimizer as: {}".format(optimizer_name))

    SAVE_FILE[0] = model_name + "_" + optimizer_name + "_out.pth"

    last_val_iterator = train_model(model, train_loader, test_loader, metric, optimizer, lr=lr,
                                    num_epochs=num_epochs, update_lr=update_lr, validation_freq=validation_freq,
                                    save=save)

    log.info("Testing the model")
    test_acc = test_model(model, test_loader, last_val_iterator)

    if save and is_verified(test_acc):
        exist_files = path_exists(ROOT_DIR + "/saved_models", SAVE_FILE[0], "contains")

        better = len(exist_files) == 0
        if not better:
            exist_acc = []
            for file in exist_files:
                exist_acc.append(float(file.split("_")[0].replace(",", ".")))
            better = all(test_acc > acc for acc in exist_acc)
        if better:
            save_model(model=model, filename=str(round(test_acc, 2)) + "_" + SAVE_FILE[0])
            saver_semaphore.acquire()

    return model


def weighted_model(model_name, pretrain_file, use_actual_num_classes=False):
    out_file = ROOT_DIR + "/saved_models/" + pretrain_file + ".pth"
    num_classes = 2 if use_actual_num_classes else 1000

    model = get_model(model_name, True, False, num_classes)

    try:
        log.info("Using class size as: {}".format(num_classes))
        return load_model(model, out_file)
    except RuntimeError as re:
        log.error(re)
        return weighted_model(model_name, pretrain_file, not use_actual_num_classes)

