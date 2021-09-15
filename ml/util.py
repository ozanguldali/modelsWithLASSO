import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from rpy2.robjects import r

from ml.helper import get_best, get_prediction, get_best_cv, get_prediction_kf, test_without_train

from util.R_util import convert_object_to_matrix, convert_list_to_floatVector, activate_robjects, install_package

from ml.save import save_model
from util.logger_util import log


def run_svm(seed, X_train, X_test, y_train, y_test, lasso, kf, lambdas=None, validate_cv=False, save=False):
    if lasso is None:
        run_svm(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=False, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)
        run_svm(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=True, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)
    else:
        log.info("Running ML model: svm")
        log.info("LASSO Enabled: " + str(lasso))

        if lasso:
            model = LinearSVC()
            grad_dict = {
                'model': [model],
                'model__loss': ['squared_hinge'],
                'model__C': lambdas,
                'model__penalty': ['l1'],
                'model__fit_intercept': [True],
                'model__dual': [False],
                'model__random_state': [seed],
                'model__max_iter': [model.max_iter ** 2]
            }
            if validate_cv:
                log.info("Cross Validation on train data")
                best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)
                C = best_params.C
                log.info("best params:\nC: {}".format(C))

                log.info("Resulting on train set:\n")
                get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
                log.info("Resulting on test set:\n")
                get_prediction(best_model, X_train, X_test, y_train, y_test)

            log.info("Grid Search and Test the model")
            best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)
            C = best_params["C"]
            log.info("best params:\nC: {}".format(C))
            model.set_params(**best_params)

        else:
            model = SVC()
            grad_dict = {
                'model': [model],
                'model__kernel': ["linear", "sigmoid", "rbf"],
                # 'model__kernel': ["sigmoid"],
                'model__C': lambdas,
                # 'model__gamma': ['scale', 'auto'],
                'model__gamma': ['scale'],
                'model__probability': [True],
                'model__random_state': [seed]
            }
            if validate_cv:
                log.info("Cross Validation on train data")
                best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)

                kernel, C, gamma = best_params.kernel, best_params.C, best_params.gamma
                log.info("best params:\nkernel: {}\tC: {}\tgamma: {}".format(kernel, C, gamma))

                log.info("Resulting on train set:\n")
                get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
                log.info("Resulting on test set:\n")
                get_prediction(best_model, X_train, X_test, y_train, y_test)

            log.info("Grid Search and Test the model")
            best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)
            kernel, C, gamma = best_params["kernel"], best_params["C"], best_params["gamma"]
            log.info("best params:\nkernel: {}\tC: {}\tgamma: {}".format(kernel, C, gamma))
            model.set_params(**best_params)

        result = get_prediction(model, X_train, X_test, y_train, y_test)
        log.info("")
        if save:
            save_model(result["classifier"],
                       str(round(float(result["acc"]), 2)) + "_lasso" if lasso else "" + "_SVM_out.joblib")


def run_lr(seed, X_train, X_test, y_train, y_test, lasso, kf, lambdas=None, validate_cv=False, save=False):
    if lasso is None:
        run_lr(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=False, kf=kf,
               lambdas=lambdas, validate_cv=validate_cv, save=save)
        run_lr(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso="l2", kf=kf, lambdas=lambdas,
               validate_cv=validate_cv, save=save)
        run_lr(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=True, kf=kf, lambdas=lambdas,
               validate_cv=validate_cv, save=save)
    else:
        log.info("Running ML model: lr")
        log.info("Penalty: " + str("LASSO" if lasso is True else lasso))

        if lasso == "l2":
            ## L2 penalty
            model = LogisticRegression()

            grad_dict = {
                'model': [model],
                'model__solver': ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear'],
                'model__penalty': ['l2'],
                'model__C': lambdas,
                'model__fit_intercept': [True],
                'model__dual': [True],
                'model__random_state': [seed],
                'model__max_iter': [model.max_iter ** 2]
            }
            if validate_cv:
                log.info("Cross Validation on train data")
                best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)

                C, solver, dual = best_params.C, best_params.solver, best_params.dual
                log.info(
                    "best params:\nC:{}\tsolver: {}\tdual: {}".format(C, solver, dual))

                log.info("Resulting on train set:\n")
                get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
                log.info("Resulting on test set:\n")
                get_prediction(best_model, X_train, X_test, y_train, y_test)

            log.info("Grid Search and Test the model")
            best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)
            C, solver, dual = best_params["C"], best_params["solver"], best_params["dual"]
            log.info(
                "best params:\nC:{}\tsolver: {}\tdual: {}".format(C, solver, dual))
            model.set_params(**best_params)

        elif lasso:
            model = LogisticRegression()
            grad_dict = {
                'model': [model],
                'model__solver': ['liblinear', 'saga'],
                'model__penalty': ['l1'],
                'model__C': lambdas,
                'model__fit_intercept': [True],
                'model__random_state': [seed],
                'model__max_iter': [model.max_iter ** 2]
            }
            if validate_cv:
                log.info("Cross Validation on train data")
                best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)

                C, solver = best_params.C, best_params.solver
                log.info("best params:\nC: {}\tsolver: {}".format(C, solver))

                log.info("Resulting on train set:\n")
                get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
                log.info("Resulting on test set:\n")
                get_prediction(best_model, X_train, X_test, y_train, y_test)

            log.info("Grid Search and Test the model")
            best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)
            C, solver = best_params["C"], best_params["solver"]
            log.info("best params:\nC: {}\tsolver: {}".format(C, solver))
            model.set_params(**best_params)

        else:
            ## no penalty
            model = LogisticRegression()
            grad_dict = {
                'model': [model],
                'model__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                'model__penalty': ['none'],
                'model__fit_intercept': [True],
                'model__random_state': [seed],
                'model__max_iter': [model.max_iter ** 2]
            }
            if validate_cv:
                log.info("Cross Validation on train data")
                best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)

                solver = best_params.solver
                log.info("best params:\nsolver: {}".format(solver))

                log.info("Resulting on train set:\n")
                get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
                log.info("Resulting on test set:\n")
                get_prediction(best_model, X_train, X_test, y_train, y_test)

            log.info("Grid Search and Test the model")
            best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)
            solver = best_params["solver"]
            log.info("best params:\nsolver: {}".format(solver))
            model.set_params(**best_params)

        result = get_prediction(model, X_train, X_test, y_train, y_test)
        log.info("")
        if save:
            save_model(result["model"],
                       str(round(float(result["acc"]), 2)) + "_l2" if lasso == "l2" else (
                                                                                             "_lasso" if lasso else "") + "_LR_out.joblib")


def run_knn(seed, X_train, X_test, y_train, y_test, kf, lasso=False, validate_cv=False, save=False):
    if lasso is None or lasso:
        log.info("Penalty approach is not used on KNN. Thus the run is performing with Penalty: False")
        run_knn(seed=seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, kf=kf, lasso=False,
                validate_cv=validate_cv, save=save)
    else:
        log.info("Running ML model: knn")
        log.info("LASSO Enabled: " + str(lasso))

        if isinstance(X_train, list):
            n_trainSamples = len(X_train)
        else:
            n_trainSamples = X_train.shape[0]

        if isinstance(X_test, list):
            n_testSamples = len(X_test)
        else:
            n_testSamples = X_test.shape[0]

        n_samples = n_trainSamples + n_testSamples

        sqrt = np.sqrt(n_samples)
        int_part = int(sqrt)
        if int_part != sqrt:
            ceil = int_part + (1 + int_part % 2)
            floor = int_part - (1 + int_part % 2)
            k_candidate = ceil if (ceil ** 2 - n_samples) < (n_samples - floor ** 2) else floor
        else:
            k_candidate = int_part

        model = KNeighborsClassifier()
        grad_dict = {
            'model': [model],
            'model__n_neighbors': [3, 5, k_candidate],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['ball_tree', 'kd_tree', 'brute'],
            # 'model__algorithm': ['auto'],
            'model__metric': ['euclidean', 'manhattan', 'chebyshev'],
        }
        if validate_cv:
            log.info("Cross Validation on train data")
            best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)

            n_neighbors, weights, algorithm, metric = best_params.n_neighbors, best_params.weights, best_params.algorithm, best_params.metric
            log.info(
                "best params:\nn_neighbors: {}\tweights: {}\talgorithm: {}\tmetric: {}".format(n_neighbors,
                                                                                               weights,
                                                                                               algorithm,
                                                                                               metric))

            log.info("Resulting on train set:\n")
            get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
            log.info("Resulting on test set:\n")
            get_prediction(best_model, X_train, X_test, y_train, y_test)

        log.info("Grid Search and Test the model")
        best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test, y_train=y_train,
                               y_test=y_test)
        n_neighbors, weights, algorithm, metric = best_params["n_neighbors"], best_params["weights"], best_params[
            "algorithm"], best_params["metric"]
        log.info(
            "best params:\nn_neighbors: {}\tweights: {}\talgorithm: {}\tmetric: {}".format(n_neighbors, weights,
                                                                                           algorithm, metric))
        model.set_params(**best_params)

        result = get_prediction(model, X_train, X_test, y_train, y_test)
        log.info("")
        if save:
            save_model(result["model"],
                       str(round(float(result["acc"]), 2)) + "_KNN_out.joblib")


def run_lda(seed, X_train, X_test, y_train, y_test, lasso, kf, lambdas=None, validate_cv=False, save=False):
    if lasso is None:
        run_lda(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=False, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)
        run_lda(seed, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lasso=True, kf=kf,
                lambdas=lambdas, validate_cv=validate_cv, save=save)
    else:
        log.info("Running ML model: lda")
        log.info("LASSO Enabled: " + str(lasso))

        if lasso:
            # install_packages(pkgs=["ps", "processx", "callr", "prettyunits", "backports", "desc", "pkgbuild", "rprojroot", "rstudioapi", "numDeriv", "SQUAREM", "evaluate", "pkgload", "praise", "colorspace", "assertthat", "utf8", "lava", "testthat", "farver", "labeling", "munsell", "R6", "RColorBrewer", "viridisLite", "cli", "crayon", "ellipsis", "fansi", "pillar", "pkgconfig", "vctrs", "stringi", "prodlim", "cpp11", "digest", "glue", "gtable", "isoband", "rlang", "scales", "tibble", "iterators", "Rcpp", "data.table", "stringr", "dplyr", "generics", "gower", "ipred", "lifecycle", "lubridate", "magrittr", "purrr", "tidyr", "tidyselect", "timeDate", "ggplot2", "foreach", "plyr", "ModelMetrics", "reshape2", "recipes", "withr", "pROC"])
            # install_package(pkg="stringr")
            # install_package(pkg="ggplot2")
            # install_package(pkg="reshape2")
            # install_package(pkg="caret")
            # install_package(pkg=ROOT_DIR.split("ml")[0]+"R_Sources/caret_6.0-86.tgz", install_type="source")
            install_package(pkg="penalizedLDA")
            install_package(pkg="TULIP")

            rDSDA = """
                function(x_tr, y_tr, x_ts, y_ts, s) {
                    test_size <- length(y_ts)
                    out <- TULIP::dsda(x=x_tr, z=NULL, y=y_tr, testx=x_ts, testz=NULL, standardize=TRUE, lambda=s, alpha=1, eps=1e-7)
                    y_pred <- out$pred
                    correct <- 0
                    for (i in 1:test_size) {
                        if (y_pred[i] == y_ts[i]) {
                            correct <- correct + 1
                        }
                    }

                    # print_value <- paste("Test Success Ratio:", paste0(100 * correct / test_size, "%"), sep = " ")
                    # return(print_value)
                    return(100 * correct / test_size)
                }
            """

            rDSDA_cv = """
                function(x, y, cv, lambdas) {
                    out <- TULIP::cv.dsda(x=x, y=y, standardize=TRUE, nfolds=cv, lambda=lambdas)
                    errors <- c(out$cvm)
                    scores <- 1.0 - errors
                    print(errors)
                    s <- out$lambda.min
                    print(paste("Best lambda:", s, sep = " "))
                    
                    out <- TULIP::cv.dsda(x=x, y=y, standardize=TRUE, nfolds=cv, lambda=c(s))
                    error <- c(out$cvm)
                    score <- 1.0 - error
                    print(paste("Best Score:", paste0(100 * score, "%"), sep = " "))
                    return(s)
                }
            """

            activate_robjects()

            if isinstance(X_train, list):
                X_train = np.array(X_train)

            if isinstance(X_test, list):
                X_test = np.array(X_test)

            tmp_y_train = [0] * len(y_train)
            for c, label in enumerate(y_train):
                if label == 0:
                    updated = 1

                else:
                    updated = 2

                tmp_y_train[c] = updated

            tmp_y_test = [0] * len(y_test)
            for c, label in enumerate(y_test):
                if label == 0:
                    updated = 1

                else:
                    updated = 2

                tmp_y_test[c] = updated

            # with a in-file script
            r_dsda = r(rDSDA)
            r_dsda_cv = r(rDSDA_cv)

            # X_train, X_test = standardize(X_train, X_test)
            rX_train = convert_object_to_matrix(X_train)
            rX_test = convert_object_to_matrix(X_test)
            ry_train = convert_list_to_floatVector(tmp_y_train)
            ry_test = convert_list_to_floatVector(tmp_y_test)

            if validate_cv:
                cv = kf.get_n_splits(X_train) if isinstance(kf, LeaveOneOut) else kf.n_splits

                log.info("Cross Validation on train data")
                out_cv = r_dsda_cv(x=rX_train, y=ry_train, cv=cv, lambdas=convert_list_to_floatVector(lambdas))
                s_cv = float(out_cv[0])
                log.info("Resulting on test set:")
                out_cv = r_dsda(x_tr=rX_train, y_tr=ry_train, x_ts=rX_test, y_ts=ry_test,
                                s=convert_list_to_floatVector([s_cv]))
                result_cv = float(out_cv[0])
                log.info("Test Success Ratio: " + str(result_cv) + "%")

            log.info("Grid Search and Test the model")
            best_score = 0.0
            best_lambda = None
            for s in lambdas:
                out = r_dsda(x_tr=rX_train, y_tr=ry_train, x_ts=rX_test, y_ts=ry_test,
                             s=convert_list_to_floatVector([s]))
                result = float(out[0])

                # save if best
                if result > best_score:
                    best_score = result
                    best_lambda = s

            log.info("Best Score: %0.6f" % best_score)
            log.info("Best Lambda: " + str(best_lambda))

            log.info("Test Success Ratio: " + str(best_score) + "%")
            log.info("")

        else:
            model = LinearDiscriminantAnalysis()
            grad_dict = {
                'model': [model],
                'model__solver': ['svd', 'lsqr', 'eigen'],
                'model__store_covariance': [True, False]
            }
            if validate_cv:
                log.info("Cross Validation on train data")
                best_params, best_model = get_best_cv(grad_dict=grad_dict, cv=kf, X=X_train, y=y_train)

                solver, store_covariance = best_params.solver, best_params.store_covariance
                log.info("best params:\nsolver: {}\tstore_covariance: {}".format(solver, store_covariance))

                log.info("Resulting on train set:\n")
                get_prediction_kf(kf=kf, classifier=best_model, X=X_train, y=y_train)
                log.info("Resulting on test set:\n")
                get_prediction(best_model, X_train, X_test, y_train, y_test)

            log.info("Grid Search and Test the model")
            best_params = get_best(grad_dict=grad_dict, X_train=X_train, X_test=X_test,
                                   y_train=y_train, y_test=y_test)
            solver, store_covariance = best_params["solver"], best_params["store_covariance"]
            log.info("best params:\nsolver: {}\tstore_covariance: {}".format(solver, store_covariance))
            model.set_params(**best_params)

            result = get_prediction(model, X_train, X_test, y_train, y_test)
            log.info("")
            if save:
                save_model(result["model"],
                           str(round(float(result["acc"]), 2)) + "_LDA_out.joblib")
