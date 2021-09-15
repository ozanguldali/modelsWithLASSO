import csv, os
from random import randrange, shuffle
from shutil import copyfile, rmtree
from math import ceil
import numpy as np

ROOT_DIR = "/Users/ozanguldali/Documents/thesis/modelsWithLASSO/"
SOURCE_DIR = "/Users/ozanguldali/Documents/thesis/covid-chestxray-dataset/images/"
CSV_PATH = "/Users/ozanguldali/Documents/thesis/modelsWithLASSO/metadata.csv"

train_covid_19_folder = ROOT_DIR+'tmp_dataset/train/COVID-19/'
train_non_covid_19_folder = ROOT_DIR+'tmp_dataset/train/non-COVID-19/'
test_covid_19_folder = ROOT_DIR+'tmp_dataset/test/COVID-19/'
test_non_covid_19_folder = ROOT_DIR+'tmp_dataset/test/non-COVID-19/'

ignored_macos_file = ".DS_Store"

# whole dataset_unique list init
covid_chestxray_dataset = []


def dataset_investigate():
    covid_patient_ids = []
    non_covid_patient_ids = []

    # read metadata file
    with open(CSV_PATH, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            covid_chestxray_dataset.append(row)

    # filter dataset_unique for COVID-19 patients having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA" or data["view"] == "AP" or data["view"] == "AP Supine"):

            covid_patient_ids.append(data["patientid"])

    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()

    # filter dataset_unique for non-COVID-19 patients or healthy people having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" not in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA" or data["view"] == "AP" or data["view"] == "AP Supine"):

            non_covid_patient_ids.append(data["patientid"])

    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()

    # check whether there exists any common data
    common_ids = set(covid_patient_ids).intersection(set(non_covid_patient_ids))

    common_ids = list(common_ids)

    print("common ids: ", len(common_ids))

    if len(common_ids) > 0:
        # remove common id info from control group
        if len(non_covid_patient_ids) > len(covid_patient_ids):
            for common_id in common_ids:
                non_covid_patient_ids.remove(common_id)
        elif len(covid_patient_ids) > len(non_covid_patient_ids):
            for common_id in common_ids:
                covid_patient_ids.remove(common_id)
        else:
            for common_id in common_ids:
                non_covid_patient_ids.remove(common_id)


    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()
    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()


def dataset_refactor():
    def check_same_file(index):
        exists = False
        file = os.path.join(folder, filename2id[f]) + "_" + str(index)
        if os.path.exists(file + "." + f.split(".")[-1]):
            index = int(file.split("_")[-1])
            file = file[:-int(index / 10) - 1]
            index += 1
            file += str(index)
            exists = True

        return file, index, exists

    # read metadata file
    filename2id = {}
    with open(CSV_PATH, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            filename2id[row["filename"]] = row["patientid"]

    folders = [train_covid_19_folder, train_non_covid_19_folder, test_covid_19_folder, test_non_covid_19_folder]

    for folder in folders:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f != ignored_macos_file]
            if len(files) > 0:
                for f in files:
                    source = os.path.join(folder, f)
                    i = 0
                    exists = True
                    while exists:
                        destination, new_i, exists = check_same_file(i)
                        if i == new_i:
                            break
                        i = new_i

                    destination += "." + f.split(".")[-1]
                    os.rename(source, destination)


def construct_dataset(unique=False, balanced=False, reset=False, create=False):
    if reset:
        prepare_directory(train_covid_19_folder)
        prepare_directory(train_non_covid_19_folder)
        prepare_directory(test_covid_19_folder)
        prepare_directory(test_non_covid_19_folder)

    # id list and whole data list inits
    covid_dataset = []
    non_covid_dataset = []
    covid_patient_ids = []
    non_covid_patient_ids = []

    # read metadata file
    with open(CSV_PATH, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            covid_chestxray_dataset.append(row)

# ---------------------------------------------------------------------------------------------------------------------

    # filter dataset_unique for COVID-19 patients having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA" or data["view"] == "AP" or data["view"] == "AP Supine"):

            if unique:
                if data["patientid"] not in covid_patient_ids:
                    covid_patient_ids.append(data["patientid"])
                    covid_dataset.append(
                        {
                            "id": data["patientid"],
                            "sex": data["sex"],
                            "age": data["age"],
                            "finding": "COVID-19",
                            "fileName": data["filename"]
                        }
                    )
            else:
                covid_patient_ids.append(data["patientid"])
                covid_dataset.append(
                    {
                        "id": data["patientid"],
                        "sex": data["sex"],
                        "age": data["age"],
                        "finding": "COVID-19",
                        "fileName": data["filename"]
                    }
                )

    # filter dataset_unique for non-COVID-19 patients or healthy people having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" not in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA" or data["view"] == "AP" or data["view"] == "AP Supine"):

            if unique:
                if data["patientid"] not in non_covid_patient_ids:
                    non_covid_patient_ids.append(data["patientid"])
                    non_covid_dataset.append(
                        {
                            "id": data["patientid"],
                            "sex": data["sex"],
                            "age": data["age"],
                            "finding": "non-COVID-19",
                            "fileName": data["filename"]
                        }
                    )
            else:
                non_covid_patient_ids.append(data["patientid"])
                non_covid_dataset.append(
                    {
                        "id": data["patientid"],
                        "sex": data["sex"],
                        "age": data["age"],
                        "finding": "non-COVID-19",
                        "fileName": data["filename"]
                    }
                )

    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()
    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()

# ---------------------------------------------------------------------------------------------------------------------

    # check whether there exists any common data
    common_ids = set(covid_patient_ids).intersection(set(non_covid_patient_ids))

    common_ids = list(common_ids)

    if len(common_ids) > 0:
        # remove common id info from small dataset
        if len(non_covid_patient_ids) >= len(covid_patient_ids):
            for common_id in common_ids:
                non_covid_patient_ids = list(filter(lambda patient_id: patient_id != common_id, non_covid_patient_ids))
            non_covid_dataset = remove_refuse_info_list_from_list(common_ids, "id", non_covid_dataset)
        elif len(covid_patient_ids) > len(non_covid_patient_ids):
            for common_id in common_ids:
                covid_patient_ids = list(filter(lambda patient_id: patient_id != common_id, covid_patient_ids))
            covid_dataset = remove_refuse_info_list_from_list(common_ids, "id", covid_dataset)

    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()
    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()

# ---------------------------------------------------------------------------------------------------------------------

    if balanced:
        # case train: experiment group is 'larger' than control group
        if len(covid_patient_ids) > len(non_covid_patient_ids):
            # elect data from larger dataset_unique to build the balance
            covid_patient_ids, covid_dataset = elect_from_larger_dataset(small=non_covid_patient_ids,
                                                                         large=covid_patient_ids, dataset=covid_dataset)

            print("not unique covid: ", len(covid_patient_ids))
            print("unique covid: ", len(np.unique(covid_patient_ids)))

        # case 2: experiment group is 'smaller' than control group
        elif len(covid_patient_ids) < len(non_covid_patient_ids):
            # elect data from larger dataset_unique to build the balance
            non_covid_patient_ids, non_covid_dataset = elect_from_larger_dataset(small=covid_patient_ids,
                                                                                 large=non_covid_patient_ids,
                                                                                 dataset=non_covid_dataset)

            print("not unique non-covid: ", len(non_covid_patient_ids))
            print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))

        # case 3: experiment group has equal size with control group -> which is okay

# ---------------------------------------------------------------------------------------------------------------------

    # construct the dataset_unique and shuffle it
    ids = covid_patient_ids + non_covid_patient_ids

    shuffle(covid_dataset)
    shuffle(non_covid_dataset)
    dataset = covid_dataset + non_covid_dataset
    shuffle(dataset)

    # get size of train and test sets
    total_size = len(dataset)
    covid_size = len(covid_dataset)
    non_covid_size = len(non_covid_dataset)

    covid_train_size = ceil(int(covid_size * 4 / 5))
    covid_train_size = covid_train_size + 1 if covid_train_size % 2 != 0 else covid_train_size
    covid_test_size = covid_size - covid_train_size

    non_covid_train_size = ceil(int(non_covid_size * 4 / 5))
    non_covid_train_size = non_covid_train_size + 1 if non_covid_train_size % 2 != 0 else non_covid_train_size
    non_covid_test_size = non_covid_size - non_covid_train_size

    train_size = covid_train_size + non_covid_train_size # int(total_size * 4 / 5)
    test_size = total_size - train_size

    print("\ntrain set size: ", train_size)
    print("covid train set size: ", covid_train_size)
    print("non-covid train set size: ", non_covid_train_size)

    print("\ntest set size: ", test_size)
    print("covid test set size: ", covid_test_size)
    print("non-covid test set size: ", non_covid_test_size)

    # construct train and test sets
    train_ids = []
    train_set = []
    test_ids = []
    test_set = []

    covid_iter = 0
    non_covid_iter = 0
    for i in range(train_size):
        if i < covid_train_size:
            train_set.append(covid_dataset[covid_iter])
            covid_iter += 1
        else:
            train_set.append(non_covid_dataset[non_covid_iter])
            non_covid_iter += 1

    # covid_iter = 0
    # non_covid_iter = 0
    for i in range(test_size):
        if i < covid_test_size:
            test_set.append(covid_dataset[covid_iter])
            covid_iter += 1
        else:
            test_set.append(non_covid_dataset[non_covid_iter])
            non_covid_iter += 1

    print('\ntrain set:')
    for train in train_set:
        print(''.join([train["id"], ' -> ', train["fileName"], ' -> ', train["finding"]]))

    print('\ntest set:')
    for test in test_set:
        print(''.join([test["id"], ' -> ', test["fileName"], ' -> ', test["finding"]]))

    if create:
        construct_related_base_directory(train_set, train_covid_19_folder, "COVID-19")
        construct_related_base_directory(train_set, train_non_covid_19_folder, "non-COVID-19")
        construct_related_base_directory(test_set, test_covid_19_folder, "COVID-19")
        construct_related_base_directory(test_set, test_non_covid_19_folder, "non-COVID-19")


def elect_from_larger_dataset(small, large, dataset):
    elected = []
    rand = []
    rand_range = len(small)

    for _ in range(rand_range):
        r = randrange(rand_range)
        while r in rand:
            r = randrange(rand_range)
        rand.append(r)
        elected.append(large[r])

    refuse = list(set(large) - set(elected))

    dataset = remove_refuse_info_list_from_list(refuse, "id", dataset)

    return elected, dataset


def remove_refuse_info_list_from_list(refuse, key, target):
    temp_list = []
    for data in target:
        if data[key] not in refuse:
            temp_list.append(data)

    target.clear()
    target.extend(temp_list)
    temp_list.clear()

    return target


def prepare_directory(folder):
    if os.path.exists(folder):
        if len(os.listdir(folder)) != 0:
            clear_directory(folder)
    else:
        create_directory(folder)


def create_directory(folder):
    os.makedirs(folder)


def clear_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def construct_related_base_directory(dataset, folder, sub_folder):
    def check_same_file(index):
        exists = False
        file = folder + data["id"] + "_" + str(index)
        if os.path.exists(file + "." + data["fileName"].split(".")[-1]):
            index = int(file.split("_")[-1])
            file = file[:-int(index / 10) - 1]
            index += 1
            file += str(index)
            exists = True

        return file, index, exists

    for data in dataset:
        if data["finding"] == sub_folder:
            source = SOURCE_DIR + data["fileName"]
            i = 0
            exists = True
            while exists:
                destination, new_i, exists = check_same_file(i)
                if i == new_i:
                    break
                i = new_i

            destination += "." + data["fileName"].split(".")[-1]
            copyfile(source, destination)


if __name__ == '__main__':
    # construct_dataset(unique=False, balanced=False, reset=True, create=True)
    # dataset_investigate()
    dataset_refactor()
