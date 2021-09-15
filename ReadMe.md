# ISTANBUL TECHNICAL UNIVERSITY - Institute of Science and Technology
## Ozan GÜLDALİ

## Department of Mathematical Engineering - Mathematical Engineering Program

## M.Sc. THESIS

## DEEP FEATURE TRANSFER FROM DEEP LEARNING MODELS INTO MACHINE LEARNING ALGORITHMS TO CLASSIFY COVID-19 FROM CHEST X-RAY IMAGES

---

* ###### dataset may be a must for some run-configurations
    - Link to dataset: https://github.com/ozanguldali/modelsWithLASSO/blob/master/dataset
* ###### dataset_constructor.py was used to create the dataset from https://github.com/ieee8023/covid-chestxray-dataset source
* ###### image_operations.py was used to investigate the augmentations of image data
* ###### visualize_layers was used to visualize the layers of cnn models
* ###### To run only ML, only CNN or both as transfer learning, app.py file can be run with corresponding function parameters.

---

### How to Run
- **transfer_learning:** True if wanted to transfer deep features from CNN model
- **save_numpy:** True if wanted to save computed features. Default is False.
- **load_numpy:** True if wanted to use previously computed features. Default is False.
- **numpy_prefix:** Prefix for numpy feature files. Default is empty string.
- **method:** "ML" or "CNN". Required if transfer_learning is False.
- **ml_model_name:** Model name for ML. "svm", "lr", "knn", "lda", or "all". Default is empty string. Required if method is not CNN.
- **ml_features:** Type of features. "info", "cnn", or "all". Default is empty "all".
- **validate_cv:** True if wanted to apply cross-validation on train set. Default is False.
- **save_ml:** True if wanted to save ML weights. Default is False.
- **save_cnn:** True if wanted to save CNN weights. Default is False.
- **cv:** Type of cross-validation. Any positive integer or "LOO". Default is 10.
- **lasso:** Type of regularization. True, False, "l2", or None. None is used for all choices. Default is False.
- **dataset_folder:** Folder name of dataset. Default is "dataset".
- **pretrain_file:** CNN pretrained pth file name without "pth" extension. Default is None.
- **batch_size:** Size of each batch. Default is 16.
- **img_size:** Size of image dimension. Default is 227.
- **num_workers:** Number of parallel workers. Default is 2.
- **augmented:** True if wanted to augment data. Default is False.
- **cnn_model_name:** Name of CNN model. "alexnet", "resnet18", "resnet34", "resnet50", "vgg16", or "vgg19". Default is empty string. Required if method is not ML.
- **optimizer_name:** Name of optimizer on CNN process. "Adam", "AdamW" or "SGD". Default is "Adam".
- **validation_freq:** Validation frequency ratio on CNN process. Any positive rational number. Default is 0.02.
- **lr:** Learning rate for CNN optimizers. Any positive rational number. Default is 0.00001.
- **momentum:** Momentum ratio for SGD optimizer. Any positive rational number. Default is 0.9.
- **weight_decay:** Weight decay ratio for Adam and AdamW. Any positive rational number. Default is 0.0001.
- **update_lr:** True if wanted to periodically decrease the learning rate on CNN process. Default is False.
- **is_pre_trained:** True if the CNN model wanted to use is pretrained. Default is True.
- **fine_tune:** True if wanted to freeze first convolution block on CNN models. Default is False.
- **num_epochs:** Number of epochs for CNN process. Any positive integer. Default is 50.       
- **normalize:** True if wanted to normalize the data. Default is True.
- **lambdas:** Lambda values for regularization on ML process. Single positive real number, or a list of positive real numbers. Default is None. List of [0.00005, 0.0001, 0.0002, 0.0005, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0] is used, if it is None.
- **seed:** System seed value for ML process. Any positive integer. Default is 4.

_Example of Transfer Leaning:_
1. (Dataset folder is required: https://github.com/ozanguldali/modelsWithLASSO/blob/master/dataset)
- Unless exists, 92.16_resnet50_Adam_out.pth file must be downloaded and inserted into "cnn/saved_models" directory.
- Link to file:
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/cnn/saved_models/92.16_resnet50_Adam_out.pth
    
`app.main(transfer_learning=True, ml_model_name="all", ml_features="all", cnn_model_name="resnet50", is_pre_trained=True,
         cv=10, dataset_folder="dataset", pretrain_file="92.16_resnet50_Adam_out", seed=4)`
  
2. (Dataset folder is not needed)
- Unless exists, 92.16_resnet50_Adam_final_X_cnn_train.npy, 92.16_resnet50_Adam_final_X_cnn_test.npy, X_info_train.npy, X_info_test.npy, y_train.npy, and y_test.npy files must be downloaded and inserted into project root directory.
- Link to files: 
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/92.16_resnet50_Adam_final_X_cnn_train.npy
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/92.16_resnet50_Adam_final_X_cnn_test.npy
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/X_info_train.npy
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/X_info_test.npy
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/y_train.npy
  - https://github.com/ozanguldali/modelsWithLASSO/blob/master/y_test.npy
    
`app.main(transfer_learning=True, ml_model_name="all", ml_features="all", load_numpy=True, validate_cv=True, cv=10,
         numpy_prefix="92.16_resnet50_Adam_final", seed=4)`

---

### Package Versions
- Python Language: 3.7.6
- Clang: 4.0.1
- pip: 20.1.1
- numpy: 1.19.0


- PyTorch: 1.5.0
- TorchSummary: 1.5.1
- TorchVision 0.6.0


- Scikit-Learn: 0.23.2


- R Language: 4.0.3
- TULIP: 1.0.1


- TensorFlow: 2.3.1
- TensorFlow-Addons: 0.11.2
- TensorFlow-Estimator: 2.3.0
- TensorFlow-Hub: 0.10.0
- TensorFlow-Probability: 0.10.0


- log4p: 2019.7.13.3
- log4python: 0.2.31
