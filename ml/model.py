import sys

from ml.util import run_svm, run_lr, run_knn, run_lda

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, X_train, X_test, y_train, y_test, seed, lasso, lambdas, kf, validate_cv=False, save=False):
    collect_garbage()

    if model_name == "svm":
        run_svm(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=lasso, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)

    elif model_name == "lr":
        run_lr(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=lasso, kf=kf,
               lambdas=lambdas, validate_cv=validate_cv, save=save)

    elif model_name == "knn":
        run_knn(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, kf=kf,
                validate_cv=validate_cv, save=save)

    elif model_name == "lda":
        run_lda(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=lasso, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)

    elif model_name == "all":
        ## SVM
        run_svm(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=lasso, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)

        ## LR
        run_lr(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=lasso, kf=kf,
               lambdas=lambdas, validate_cv=validate_cv, save=save)

        ## KNN
        run_knn(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, kf=kf,
                validate_cv=validate_cv, save=save)

        ## LDA
        run_lda(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=lasso, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)

    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)
