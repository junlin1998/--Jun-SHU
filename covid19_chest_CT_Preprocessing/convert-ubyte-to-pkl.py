import gzip
import pickle
import os
import numpy as np
from PIL import Image


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    print(file_path)
    if os.path.exists(file_path):
        return
    else:
        print("no datasets available")

def download_mnist():
    for v in key_file.values():
       _download(v)

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)# 本身是8
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)#本身是16
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = []
    dataset.append(_load_label(key_file['train_label']))
    dataset.append(_load_img(key_file['train_img']))
    dataset.append(_load_label(key_file['test_label']))
    dataset.append(_load_img(key_file['test_img']))
    return dataset

def ubyte_to_pkl():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file for traning data...")
    with open(save_file_1, 'wb') as f:
        pickle.dump([dataset[0],dataset[1]], f, -1)

    print("Creating pickle file for test data...")
    with open(save_file_2, 'wb') as f:
        pickle.dump([dataset[2],dataset[3]], f, -1)
    print("Done")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_mnist(dtype, normalize=True, flatten=True, one_hot_label=False):
    """
    Parameters
    ----------
    normalize : Normalize the pixel values
    flatten : Flatten the images as one array
    one_hot_label : Encode the labels as a one-hot array

    Returns
    -------
    (Trainig Image, Training Label), (Test Image, Test Label)
    """
    save_file = ''
    if dtype == 'training':
        save_file = save_file_1
    elif dtype == 'test':
        save_file = save_file_2
    if not os.path.exists(save_file):
        ubyte_to_pkl()

    with open(save_file_1, 'rb') as f:
        l_dataset = pickle.load(f)
    # if normalize:
    #     for key in ('train_img', 'test_img'):
    #         dataset[key] = dataset[key].astype(np.float32)
    #         dataset[key] /= 255.0
    #
    # if not flatten:
    #      for key in ('train_img', 'test_img'):
    #         dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    #
    # if one_hot_label:
    #     dataset[1] = _change_one_hot_label(dataset[1])
    #     dataset[3] = _change_one_hot_label(dataset[3])

    return (l_dataset[0],l_dataset[-1])

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# Load the MNIST dataset
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'test-images-idx3-ubyte.gz',
    'test_label':'test-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file_1 = dataset_dir + "/covid19_train.pkl"
save_file_2 = dataset_dir + "/covid19_test.pkl"

img_dim = (1, 64, 64)
img_size = 4096

x_train, t_train = load_mnist("train")
x_test, t_test = load_mnist("test")

# Show the sample image
label = x_train[2]
img = t_train[2]
print(label)

print(img.shape)
img = img.reshape(64, 64)
print(img.shape)

img_show(img)