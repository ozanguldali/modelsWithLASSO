from torchsummary import summary

from cnn import device

from util.logger_util import log


def get_summary(model, train_loader):
    im_shape = train_loader.__iter__().next()[0].shape
    summary(model, im_shape[1:])


def get_fine_tuned_summary(convolutional, classifier, train_loader):
    log.info("Frozen Convolutional Block:")
    im = train_loader.__iter__().next()[0]
    summary(convolutional, im.shape[1:])

    log.info("\n")
    im = im.to(device)

    log.info("Classification Block:")
    im_shape = convolutional(im).shape
    summary(classifier, im_shape[1:])
