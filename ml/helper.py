import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, ParameterGrid
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

from ml import ROOT_DIR
from ml.dataset import read_dataset, divide_dataset, standardize
from util.file_util import path_exists

from util.logger_util import log


def get_prediction_kf(kf, classifier, X, y):
    cv_logger = "LOO" if isinstance(kf, LeaveOneOut) else str(kf.n_splits) + "-Fold CV"
    ratios = []
    conf_matrices = []
    roc_list = []
    for e, (train, test) in enumerate(tqdm(kf.split(X, y))):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # log.info("iteration - {}".format(e))
        X_train, X_test, y_train, y_test = X[train], X[test], np.array(y)[train], np.array(y)[test]
        X_train, X_test = standardize(X_train, X_test)

        # log.info("prepared data")
        classifier.fit(X_train, y_train)
        # log.info("model has been fitted")
        success_ratio = classifier.score(X_test, y_test)
        # log.info("got the score")
        # log.info(cv_logger + " -- Iteration " + str(e) + " Test Success Ratio: " + str(100*success_ratio) + "%")
        ratios.append(success_ratio)

        classifier.predict(X_test)

        y_hat = classifier.predict(X_test)
        auc = roc_auc_score(y_test, y_hat)
        # log.info(cv_logger + " -- Iteration " + str(e) + " AUC Score: " + str(auc))
        roc_list.append(auc)

        conf_matrix = confusion_matrix(y_test.tolist(), classifier.predict(X_test).tolist())
        # log.info(cv_logger + " -- Iteration " + str(e) + " Confusion Matrix:\n" + str(conf_matrix))
        conf_matrices.append(conf_matrix)

    log.info(cv_logger + " Average Test Success Ratio: " + str(100 * np.average(np.array(ratios))) + "%")
    log.info(cv_logger + " Average AUC Score: " + str(np.average(np.array(roc_list))))
    log.info(cv_logger + " Average Confusion Matrix:\n" + str(np.mean(conf_matrices, axis=0)))

    return {"classifier": classifier, "acc": str(100 * np.average(np.array(ratios)))}


def get_prediction(classifier, X_train, X_test, y_train, y_test, scalar=True):
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    if scalar:
        X_train, X_test = standardize(X_train, X_test)
    # log.info("prepared data")
    classifier.fit(X_train, y_train)
    # log.info("model has been fitted")

    success_ratio = classifier.score(X_test, y_test)
    log.info("Test Success Ratio: " + str(100 * success_ratio) + "%")

    y_hat = classifier.predict(X_test)
    auc = roc_auc_score(y_test, y_hat)
    log.info("Average AUC Score: " + str(auc))

    y_hat = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test.tolist(), y_hat.tolist())
    log.info("Average Confusion Matrix:\n" + str(conf_matrix))

    return {"classifier": classifier, "acc": str(100 * success_ratio)}


def test_without_train(classifier, X_train, X_test, y_test):
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train, X_test = standardize(X_train, X_test)
    # log.info("prepared data")
    # classifier.fit(X_train, y_train)
    # log.info("model has been fitted")

    success_ratio = classifier.score(X_test, y_test)
    log.info("Test Success Ratio: " + str(100 * success_ratio) + "%")

    # test_prob = classifier._predict_proba_lr(X_test) if isinstance(classifier, LinearSVC) else classifier.predict_proba(X_test)
    # roc_score = roc_auc_score(y_test, test_prob)
    # log.info("Average AUC Score: " + str(roc_score))

    y_hat = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test.tolist(), y_hat.tolist())
    log.info("Average Confusion Matrix:\n" + str(conf_matrix))

    return {"classifier": classifier, "acc": str(100 * success_ratio)}


def get_dataset(model_name, dataset_folder, img_size, normalize, divide=False):
    X_file, y_file = os.path.join(ROOT_DIR, "X.npy"), os.path.join(ROOT_DIR, "y.npy")

    if not (path_exists(folder=ROOT_DIR, file=X_file) and path_exists(folder=ROOT_DIR, file=y_file)):
        log.info("Reading dataset")
        X, y = read_dataset(dataset_folder=dataset_folder, resize_value=(img_size, img_size), to_crop=True)
        np.save(X_file, X)
        np.save(y_file, y)
    else:
        log.info("Loading dataset")
        X, y = np.load(X_file), np.load(y_file)

    if normalize:
        X = StandardScaler(with_mean=True).fit_transform(X)

    if model_name == "lda":
        for i, label in enumerate(y):
            if label <= 0:
                updated = 1

            else:
                updated = 2

            y[i] = updated

    if divide:
        log.info("Dividing dataset into train and test data")
        X_tr, y_tr, X_ts, y_ts = divide_dataset(X, y)
        log.info("Train data length: %d" % len(y_tr))
        log.info("Test data length: %d" % len(y_ts))

        return X_tr, y_tr, X_ts, y_ts

    return X, y


def get_best(grad_dict, X_train, X_test, y_train, y_test, model=None):
    X_train, X_test = standardize(X_train, X_test)

    if model is None:
        model = grad_dict['model'][0]
    del grad_dict['model']

    best_score = 0.0
    scores = []
    best_grid = []
    for g in tqdm(ParameterGrid(grad_dict)):
        keys = list(g.keys()).copy()
        for key in keys:
            if 'model__' in key:
                g[key.replace('model__', '')] = g.pop(key)
        model.set_params(**g)

        try:
            model.fit(X_train, y_train)
            success_ratio = model.score(X_test, y_test)
            scores.append(success_ratio)
            # save if best
            if success_ratio > best_score:
                best_score = success_ratio
                best_grid = g
        except ValueError:
            continue
        except np.linalg.LinAlgError as err:
            if isinstance(model, LinearDiscriminantAnalysis):
                continue
            else:
                raise err

    print(np.array(scores))
    log.info("Best Score: %0.6f" % best_score)
    log.info("Best Grid:" + str(best_grid))

    return best_grid


def get_best_cv(grad_dict, cv, X, y, model=None):
    if model is None:
        model = grad_dict['model'][0]

    pipe = Pipeline([('scale', StandardScaler(with_mean=True)), ('model', model)])

    param_grid = [grad_dict]

    clf = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=1, n_jobs=8, pre_dispatch=2*8)

    fitted = clf.fit(X, y)
    best_estimator = fitted.best_estimator_.named_steps.model

    print(fitted.cv_results_["mean_test_score"])
    print(fitted.best_score_)
    # print(fitted.best_params_['model'].random_state)
    # breakpoint()
    return fitted.best_params_['model'], best_estimator
