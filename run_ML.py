from sklearn.model_selection import KFold, LeaveOneOut

from ml.helper import get_dataset
from ml.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log


def main(seed, model_name, dataset_folder="dataset", cv: object = 10, lasso: object = False, img_size=227, normalize=True, lambdas=None,
         save=False):

    if lambdas is None:
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    if cv in ["loo", -1]:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    log.info("Constructing datasets and arrays")
    X, y = get_dataset(model_name, dataset_folder, img_size, normalize, divide=False)

    log.info("Calling the model: " + model_name)
    run_model(model_name=model_name, X=X, y=y, seed=seed, lasso=lasso, kf=kf, lambdas=lambdas, save=save)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(seed=4, model_name='svm', cv=5, lasso=False)
    log.info("Process Finished")
