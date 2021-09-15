import torch

from cnn import ROOT_DIR, saver_semaphore


def save_model(model, filename):
    torch.save(model.state_dict(), ROOT_DIR + "/saved_models/" + filename)
    saver_semaphore.release()

