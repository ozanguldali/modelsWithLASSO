import sys

from cnn import device
from cnn.helper import set_dataset_and_loaders
from cnn.model import run_model, weighted_model
from cnn.test import test_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", batch_size=64, img_size=227, test_without_train=False, pretrain_file=None,
         num_workers=2, model_name='alexnet', optimizer_name='Adam', is_pre_trained=False, fine_tune=False, augmented=False,
         num_epochs=200, update_lr=False, normalize=None, validation_freq=0.05, lr=0.0001, momentum=0.9, weight_decay=1e-4):

    if test_without_train and pretrain_file is None:
        log.fatal("Pretrained weight file is a must on test without train approach.")
        sys.exit(1)

    if not is_pre_trained and fine_tune:
        fine_tune = False

    log.info("Constructing datasets and loaders")
    train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder=dataset_folder,
                                                                               augmented=augmented,
                                                                               batch_size=batch_size,
                                                                               img_size=img_size,
                                                                               num_workers=num_workers,
                                                                               shuffle_train=True,
                                                                               normalize=normalize)

    log.info(test_loader.dataset.class_to_idx)

    log.info("Calling the model: " + model_name)
    if test_without_train:
        model = weighted_model(model_name, pretrain_file, True)
        model = model.to(device)
        test_model(model, test_loader, 0)

    else:
        run_model(model_name=model_name, optimizer_name=optimizer_name, is_pre_trained=is_pre_trained,
                  fine_tune=fine_tune, train_loader=train_loader, test_loader=test_loader,
                  num_epochs=num_epochs, save=save, update_lr=update_lr, validation_freq=validation_freq,
                  lr=lr, momentum=momentum, weight_decay=weight_decay)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    # main(test_without_train=True, model_name="resnet50", is_pre_trained=True, pretrain_file="94.12_resnet50_Adam_out")
    main(save=False, model_name="vgg16", optimizer_name="Adam", is_pre_trained=True, batch_size=16, lr=0.00001,
         num_epochs=1, validation_freq=1/10, augmented=False, dataset_folder="dataset")
    log.info("Process Finished")
