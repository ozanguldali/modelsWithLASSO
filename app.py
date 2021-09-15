import os
import sys
from sklearn.model_selection import LeaveOneOut, KFold

import run_ML
from ml.model import run_model
from ml.dataset import normalize as normalize_data

import run_CNN
from cnn import model as cnn_model, device
from cnn.dataset import set_age_sex, set_loader
from cnn.features import extract_features, feature_clean_withinclass, feature_clean
from cnn.helper import set_dataset_and_loaders, get_feature_extractor
from util.file_util import path_exists

from util.garbage_util import collect_garbage
from util.logger_util import log

import numpy as np

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))


def main(transfer_learning, save_numpy=False, load_numpy=False, numpy_prefix="", method="", ml_model_name="",
         ml_features="all", validate_cv=False, save_ml=False, save_cnn=False,
         cv: object = 10, lasso: object = False, dataset_folder="dataset", pretrain_file=None, batch_size=16,
         img_size=227,
         num_workers=2, augmented=False, cnn_model_name="", optimizer_name='Adam', validation_freq=0.02,
         lr=0.00001, momentum=0.9, weight_decay=1e-4, update_lr=False, is_pre_trained=True, fine_tune=False,
         num_epochs=50, normalize=True, lambdas=None, seed=1):
    if lambdas is None:
        lambdas = [0.00005, 0.0001, 0.0002, 0.0005, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

    if save_numpy or load_numpy:
        X_info_train_file = "X_info_train.npy"
        X_info_test_file = "X_info_test.npy"
        X_cnn_train_file = "_X_cnn_train.npy"
        X_cnn_test_file = "_X_cnn_test.npy"
        y_train_file = "y_train.npy"
        y_test_file = "y_test.npy"
        X_cnn_train_file = numpy_prefix + X_cnn_train_file
        X_cnn_test_file = numpy_prefix + X_cnn_test_file

    if not transfer_learning:
        if method.lower() == "ml":
            run_ML.main(model_name=ml_model_name, dataset_folder=dataset_folder, seed=seed, cv=cv,
                        img_size=img_size, normalize=normalize, lambdas=lambdas, lasso=lasso, save=save_ml)
        elif method.lower() == "cnn":
            run_CNN.main(save=save_cnn, dataset_folder=dataset_folder, batch_size=batch_size, test_without_train=False,
                         img_size=img_size, num_workers=num_workers, num_epochs=num_epochs, model_name=cnn_model_name,
                         optimizer_name=optimizer_name, is_pre_trained=is_pre_trained, pretrain_file=None,
                         augmented=augmented,
                         fine_tune=fine_tune, update_lr=update_lr, normalize=normalize, validation_freq=validation_freq,
                         lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            log.fatal("method name is not known: " + method)
            sys.exit(1)

    else:

        if load_numpy:
            log.info("Loading from: " + X_info_train_file)
            X_info_train = np.load(X_info_train_file)
            log.info("Loading from: " + X_info_test_file)
            X_info_test = np.load(X_info_test_file)

            log.info("Loading from: " + X_cnn_train_file)
            X_cnn_train = np.load(X_cnn_train_file)
            log.info("Loading from: " + X_cnn_test_file)
            X_cnn_test = np.load(X_cnn_test_file)

            log.info("Loading from: " + y_train_file)
            y_train = list(np.load(y_train_file))
            log.info("Loading from: " + y_test_file)
            y_test = list(np.load(y_test_file))

        else:
            load_cnn_features = is_pre_trained and pretrain_file is not None and cnn_model_name in pretrain_file.lower()
            shuffle_train = not load_cnn_features if transfer_learning else True

            log.info("Constructing datasets and loaders")
            train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder=dataset_folder,
                                                                                       augmented=augmented,
                                                                                       batch_size=batch_size,
                                                                                       img_size=img_size,
                                                                                       num_workers=num_workers,
                                                                                       normalize=normalize,
                                                                                       shuffle_train=shuffle_train)
            y_train = train_data.targets
            y_test = test_data.targets
            class0, class1 = 0, 1
            class_dist = {i: (y_train + y_test).count(i) for i in (y_train + y_test)}
            class0_size = class_dist[class0]
            class1_size = class_dist[class1]
            log.info("Total class 0 size: " + str(class0_size))
            log.info("Total class 1 size: " + str(class1_size))

            if ml_features != "info":
                if load_cnn_features:
                    log.info(
                        "Getting PreTrained CNN model: " + cnn_model_name + " from the Weights of " + pretrain_file)
                    model = cnn_model.weighted_model(cnn_model_name, pretrain_file, True)

                else:
                    log.info("Running CNN model: " + cnn_model_name)
                    model = cnn_model.run_model(model_name=cnn_model_name, optimizer_name=optimizer_name,
                                                is_pre_trained=is_pre_trained,
                                                fine_tune=fine_tune, train_loader=train_loader, test_loader=test_loader,
                                                num_epochs=num_epochs, save=save_cnn, update_lr=update_lr,
                                                validation_freq=validation_freq,
                                                lr=lr, momentum=momentum, weight_decay=weight_decay)

                log.info("Feature extractor is being created")
                feature_extractor = get_feature_extractor(cnn_model_name, model.eval())
                log.info("Feature extractor is setting to device: " + str(device))
                feature_extractor = feature_extractor.to(device)

                log.info("Extracting features as X_cnn array")
                X_cnn_train = extract_features(train_loader, feature_extractor)
                X_cnn_test = extract_features(test_loader, feature_extractor)

                log.info("Number of features in X_cnn: " + str(len(X_cnn_test[0])))

                if normalize:
                    log.info("Normalizing the X cnn feature matrix")
                    X_cnn_train, X_cnn_test = normalize_data(X_cnn_train, X_cnn_test, y_train, y_test)

            if ml_features != "cnn":
                log.info("Pulling age and sex lists from metadata file")
                age_list_train, sex_list_train = set_age_sex(dataset=train_data)
                age_list_test, sex_list_test = set_age_sex(dataset=test_data)

                log.info("Creating X_info array by age and sex info")
                # M: 0 and F: 1
                # X_info = list(zip(age_list, sex_list))
                X_info_train = []
                for c in range(len(train_data)):
                    X_info_train.append([age_list_train[c], sex_list_train[c]])
                X_info_test = []
                for c in range(len(test_data)):
                    X_info_test.append([age_list_test[c], sex_list_test[c]])

                log.info("Number of features in X_info: " + str(len(X_info_test[0])))

                if normalize:
                    log.info("Normalizing the X cnn feature matrix")
                    X_info_train, X_info_test = normalize_data(X_info_train, X_info_test, y_train, y_test)

            if save_numpy:
                if ml_features != "cnn":
                    if not path_exists(folder=ROOT_DIR, file=X_info_train_file):
                        log.info("Saving to: " + X_info_train_file)
                        np.save(X_info_train_file, X_info_train)
                    if not path_exists(folder=ROOT_DIR, file=X_info_test_file):
                        log.info("Saving to: " + X_info_test_file)
                        np.save(X_info_test_file, X_info_test)

                if ml_features != "info":
                    if not path_exists(folder=ROOT_DIR, file=X_cnn_train_file):
                        log.info("Saving to: " + X_cnn_train_file)
                        np.save(X_cnn_train_file, X_cnn_train)
                    if not path_exists(folder=ROOT_DIR, file=X_cnn_test_file):
                        log.info("Saving to: " + X_cnn_test_file)
                        np.save(X_cnn_test_file, X_cnn_test)

                if not path_exists(folder=ROOT_DIR, file=y_train_file):
                    log.info("Saving to: " + y_train_file)
                    np.save(y_train_file, np.array(y_train))
                if not path_exists(folder=ROOT_DIR, file=y_test_file):
                    log.info("Saving to: " + y_test_file)
                    np.save(y_test_file, np.array(y_test))

        # num_train = len(X_cnn_train)
        # X_cnn, y_cnn = [], []
        # X_cnn.extend(X_cnn_train)
        # X_cnn.extend(X_cnn_test)
        # y_cnn.extend(y_train)
        # y_cnn.extend(y_test)
        # X_cnn = feature_clean_withinclass(X_cnn, y_cnn, class0_size, class1_size, class0)
        # # X_cnn = feature_clean(X_cnn)
        # X_cnn_train = X_cnn[:num_train]
        # X_cnn_test = X_cnn[num_train:]

        if ml_features == "all":
            log.info("Creating merged and divided general X feature array")
            X_train = []
            for c in range(len(y_train)):
                row = []
                row.extend(X_info_train[c])
                row.extend(X_cnn_train[c])
                X_train.append(row)
            X_test = []
            for c in range(len(y_test)):
                row = []
                row.extend(X_info_test[c])
                row.extend(X_cnn_test[c])
                X_test.append(row)

            if normalize:
                log.info("Normalizing the X feature matrix")
                X_train, X_test = normalize_data(X_train, X_test, y_train, y_test)

            log.info("Number of features in merged X: " + str(len(X_test[0])))

        if validate_cv:
            if cv in ["loo", -1]:
                kf = LeaveOneOut()
            else:
                kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kf = None

        if ml_features == "info":
            log.info("Using X info feature matrix")
            run_model(model_name=ml_model_name, X_train=X_info_train, X_test=X_info_test, y_train=y_train,
                      y_test=y_test, seed=seed, lasso=lasso, lambdas=lambdas,
                      kf=kf, validate_cv=validate_cv, save=save_ml)
        elif ml_features == "cnn":
            log.info("Using X cnn feature matrix")
            run_model(model_name=ml_model_name, X_train=X_cnn_train, X_test=X_cnn_test, y_train=y_train, y_test=y_test,
                      seed=seed, lasso=lasso, lambdas=lambdas,
                      kf=kf, validate_cv=validate_cv, save=save_ml)
        elif ml_features == "all":
            log.info("Using X merged feature matrix")
            run_model(model_name=ml_model_name, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                      seed=seed, lasso=lasso, lambdas=lambdas,
                      kf=kf, validate_cv=validate_cv, save=save_ml)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(transfer_learning=True, ml_model_name="all", ml_features="all", load_numpy=True,
         numpy_prefix="92.16_resnet50_Adam_final", validate_cv=True, seed=4)
    log.info("Process Finished")
