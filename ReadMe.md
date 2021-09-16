# ISTANBUL TECHNICAL UNIVERSITY - Institute of Science and Technology
## Ozan GÜLDALİ

## Department of Mathematical Engineering - Mathematical Engineering Program

## M.Sc. THESIS

## DEEP FEATURE TRANSFER FROM DEEP LEARNING MODELS INTO MACHINE LEARNING ALGORITHMS TO CLASSIFY COVID-19 FROM CHEST X-RAY IMAGES

[Summary](#summary)

[Instructions](#instructions)

[How to Run](#how-to-run)

[Package Versions](#package-versions)


---

### Summary

Coronavirus disease 2019 (COVID-19) is a contagious disease caused by SARS-CoV-2. It was first reported on December 2019 in Wuhan, China, and declared as a pandemic on 11 March, 2020. Even though the disease is a severe acute respiratory illness, it affects various organs and causes several symptoms such as fever, dry cough, tiredness, the loss of taste or smell, diarrhoea, headache, aches and pains, sore throat, and conjunctivitis. As of the beginning of July 2021, over 185 million people have been infected and more than 4 million people died because of COVID-19. For that reason, one of the most important issues is the diagnosis of COVID-19. Although the most basic method to diagnose COVID-19 is Polymerase Chain Reaction (PCR) test, different techniques have been being experimented and developed. Since COVID-19 has a huge effect on lungs, diagnosis methods based on lung characteristics and images are emphasized. However, there are various illnesses affecting lungs. Hence, it has been an important challenge and problem to find a procedure to classify COVID-19 with high success rate.

In this thesis, we suggested use of deep feature transformation from deep learning models to machine learning (ML) algorithms to classify patients COVID-19 infected via chest X-rays. In addition to image data, we also used the demographic information of patients during ML process to contribute to the information coming from deep features. Chapter 1 gives background information about our problem, the purpose of our study, the related literature survey and the structure of this thesis. The basic information about our image data, chest X-rays, are given in Chapter 2.

The problem we focused on is a binary classification between COVID-19 patients and other people. In order to solve this problem, we used data set containing 131 COVID-19 and 123 non-COVID-19 labeled data. Then, we divided the data set into train and test sets so that 80% of the total data is in the train set, and then augmented the train set with horizontal flip, vertical flip, 90 degrees of rotation, 180 degrees of rotation, and 270 degrees of rotation to increase the size of the train set. Thus, at the end, we yielded 630 COVID-19 and 588 non-COVID-19 labeled data in train set, and 26 COVID-19 and 25 non-COVID-19 labeled data in test set. The augmented data was used on CNN experiments only. At the beginning of Chapter 5, the data set and augmentation technique was detailed.

The deep learning models we used are Convolutional Neural Network (CNN) models such that AlexNet, ResNet-18, ResNet-34, ResNet-50, VGG16, and VGG19. We particularly experimented three different optimization methods for each CNN model such that SGD with momentum, Adam and Adam with decoupled weight decay. The objective loss function was to minimize cross-entropy loss function which was common for each model. Each image sample was resized to 227 x 227, center cropped, converted to gray-scale, and then normalized. Chapter 3 consists of the introduction to deep learning, basic information about CNNs, and how to perform transfer learning. Two types of transfer learning were used in this study, which are transferring pre-trained model weights into CNN models and transferring deep features extracted from CNN models into ML algorithms. Pre-trained CNN models are the models that previously trained on ImageNet data set on the record, and we performed re-train after initializing the models with these recorded weights. Deep feature transfer learning is extracting the features of CNN model from the fully-connected block of model, and using it as feature matrix in another artificial intelligence technique such as ML algorithms.

The ML algorithms we used are supervised learning algorithms such that Support Vector Machines (SVM), Logistic Regression (LR), K-Nearest Neighbor (KNN) and Linear Discriminant Analysis (LDA). We experimented different regularization techniques, which are Lasso known as L1 norm and Ridge known as L2 norm, on stated ML algorithms. Chapter 4 consists of the introduction to ML, basic information about algorithms and regularizers, and cross-validation technique. We performed 10-Fold cross-validation on train set to obtain the generalized hyper-parameter choices besides hyper-parameters specific to our initially split test set. The algorithms and experiments were applied to the feature set of demographic information, the deep transferred feature set, and the combination of transferred features and demographic information separately. The demographic information feature matrix clearly consists of two feature columns such that age and sex information. The length of transferred deep features for each sample is thousand. Hence, the combined feature matrix contains thousand two columns.

All experiments for CNN and ML are detailed in Chapter 5, including data pre-processing and hyper-parameter tuning techniques for ML specifically. Grid search was used to find optimal parameters for each feature matrix and algorithm. The source code for experiments was mainly carried out in Python programming language, and a small part was done in R programming language. The CNN models were applied using PyTorch library in Python, and the ML algorithms were applied using Sklearn library in Python. Only regularized LDA algorithm was coded in R programming language using TULIP library. We performed our CNN experiments on GPU to have faster and parallel processes. Since we did not have an opportunity to reach a physical computer including GPU that we can use during our experiments, we performed the experiments on the Google Colaboratory platform. It is a partially-free platform for Gmail users to implement CUDA to use its provided GPU. After collecting CNN results and \*.pth files containing the best model weights, the ML experiments were performed locally on CPU.

We explained the performance measurement techniques in Chapter 6 together with experiment results for CNN and ML processes. The best result was achieved by using ResNet-50 model with Adam optimizer. The metrics on this result are 92.16%, 0.9216, 0.9215, 0.9216, 0.9216, 0.9219 for the accuracy, sensitivity, specificity, precision, F1 score, and AUC score respectively. Since we experimented for obtaining optimal hyper-parameters for both generalized and specific to our test data, the results for both were reported too. For the feature matrix of demographic information, the best results for both generalized and chosen test set hyper-parameters are the same, and achieved with KNN algorithm. The metrics on this result are the accuracy of 56.86%, the sensitivity of 0.5686, the specificity of 0.5837 the precision of 0.6863, the F1 score of 0.4955, and the AUC score of 0.5745. For the deep feature matrix obtained from ResNet-50 model weights, SVM with Ridge penalty, LR, LR with penalty, and KNN algorithms had the same results according to generalized hyper-parameters. The metrics on this results are the accuracy, sensitivity, specificity, precision, F1 score and AUC score of 92.16%, 0.9216, 0.9230, 0.9243, 0.9215, 0.9223 respectively. Finally, for the combined feature matrix of demographic information and extracted deep features obtained from ResNet-50 model weights, SVM with Ridge penalty, LR, LR with Ridge penalty, and KNN algorithms had the same results as well according to generalized hyper-parameters. The metrics on this results are the accuracy, sensitivity, specificity, precision, F1 score and AUC score of 92.16%, 0.9216, 0.9230, 0.9243, 0.9215, 0.9223 respectively.

In conclusion, according to stated results, we yielded an improvement of using regularization with linear discriminant analysis and Lasso regularizer. In general, we did not have an improvement by combining demographic information with deep features. However, we anticipate an improvement with this image and non-image data combining technique by using more data samples and more information about patients such as doctors report, tobacco product use, associated genetic diseases, respiratory test information, etc. Finally, even though we could not see an improvement from CNN testing results to ML testing results in terms of the accuracy, sensitivity and F1 score, the specificity and precision improved, as we discussed in Chapter 7, a data set with more samples and these samples inspected by subject matter experts, such as specialist radiologists for our X-rays, would allow the study to have better metric results and better comparison opportunities between experimental phases.

---

### Instructions

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
