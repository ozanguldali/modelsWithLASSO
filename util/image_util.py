import math

import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import cnn.helper as cnn_helper

DATASET_ROOT = "/Users/ozanguldali/Documents/thesis/covid-chestxray-dataset_unique/images/"

image_list = [
    ["train-s2.0-S1684118220300608-main.pdf-001.jpg", 1],
    ["train-s2.0-S1684118220300608-main.pdf-002.jpg", 1],
    ["train-s2.0-S1684118220300682-main.pdf-002-a1.png", -1],
    ["train-s2.0-S1684118220300682-main.pdf-002-a2.png", -1],
    ["train-s2.0-S1684118220300682-main.pdf-003-b1.png", -1],
    ["train-s2.0-S1684118220300682-main.pdf-003-b2.png", 1]
]

resize_value = (80, 80)

X = []
y = []
X_test = []

image_data = []


def load_data(cropped=False):
    #for k in range(0, 40):
    for i in range(len(image_list)):

        img_read = imread(DATASET_ROOT + image_list[i][0], as_gray=True)

        if cropped:
            (y_point, x_point) = img_read.shape
            crop_size = np.min([y_point, x_point])
            start_x = x_point // 2 - (crop_size // 2)
            start_y = y_point // 2 - (crop_size // 2)

            img_read = img_read[start_y:start_y + crop_size, start_x:start_x + crop_size]

        img_read = resize(img_read, resize_value, anti_aliasing=True)

        # print(img_read.shape)

        features = np.reshape(img_read, img_read.shape[0] * img_read.shape[1])

        # print(features.shape)

        image_data.append([features, image_list[i][1]])

        # with open("/Users/ozanguldali/Documents/thesis/covid-chestxray-dataset_ceren/images/" + image_list[i][0], "rb") as image:
        #     f = image.read()
        #     b = bytearray(f)
        #
        #     # Image.open(io.BytesIO(b)).convert("RGBA").show()
        #
        #     nparr = np.frombuffer(b, dtype=np.uint8)
        #     image_data.append([nparr, image_list[i][train]])
        #
        #     # image_data.append([cv2.imdecode(nparr, cv2.IMREAD_COLOR), image_list[i][train]])
        X.append(features.tolist())
        y.append(image_data[i][1])

# for i in range(len(image_list)):
#     X.append(image_data[i][0].tolist())
#     y.append(image_data[i][train])
#
# X_test.append(image_data[len(image_list)-train][0])

    return X, y, X_test


def view_resized_dataset(cropped=False):
    for i in range(len(image_list)):
        view_resized_data(DATASET_ROOT + image_list[i][0], cropped)


def view_resized_data(file, cropped=False):
    img_read = imread(file, as_gray=True)

    if cropped:
        (y_point, x_point) = img_read.shape
        crop_size = np.min([y_point, x_point])
        start_x = x_point // 2 - (crop_size // 2)
        start_y = y_point // 2 - (crop_size // 2)

        img_read = img_read[start_y:start_y + crop_size, start_x:start_x + crop_size]


    img_read = resize(img_read, resize_value, anti_aliasing=True)

    imshow(img_read)

    plt.tight_layout()
    plt.show()


def view_dataset(cropped=False):
    for i in range(len(image_list)):
        view_data(DATASET_ROOT + image_list[i][0], cropped)


def view_data(file, cropped=False):
    img_read = imread(file, as_gray=True)

    if cropped:
        (y_point, x_point) = img_read.shape
        crop_size = np.min([y_point, x_point])
        start_x = x_point // 2 - (crop_size // 2)
        start_y = y_point // 2 - (crop_size // 2)

        img_read = img_read[start_y:start_y + crop_size, start_x:start_x + crop_size]

    imshow(img_read)

    plt.tight_layout()
    plt.show()


def view_normalized_dataset():
    for i in range(len(X)):
        view_normalized_data(i)


def view_normalized_data(index):
    image_data = X[index]

    features_shape = image_data.shape

    dim = int(math.sqrt(features_shape[0]))

    img_read = np.reshape(image_data, (-1, dim))
    imshow(img_read)

    plt.tight_layout()
    plt.show()


def visualize():
    train_data, train_loader, test_data, test_loader = cnn_helper.set_dataset_and_loaders("dataset", False, 8,
                                                                                          227, 4, None)
    for data in test_data:
        if data[1] == 1:
            image = data[0]
            break

    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.show()


if __name__ == '__main__':
    index = 0

    X, _, _ = load_data(cropped=True)
    X = StandardScaler(with_mean=True).fit_transform(X)

    view_data(file=DATASET_ROOT + image_list[index][0], cropped=False)
    view_data(file=DATASET_ROOT + image_list[index][0], cropped=True)

    view_resized_data(file=DATASET_ROOT + image_list[index][0], cropped=True)

    view_normalized_data(index)

    # view_normalized_data(len(image_list) - index - train)

