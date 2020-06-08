# Compute superpixels for MNIST/CIFAR-10 using SLIC algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

import numpy as np
import random
import os
import scipy
import pickle
from skimage.segmentation import slic
import multiprocessing as mp
import scipy.ndimage
import scipy.spatial
import argparse
import datetime
import torch
import codecs

def parse_args():
    parser = argparse.ArgumentParser(description='Extract SLIC superpixels from images')
    parser.add_argument('-D', '--dataset', type=str, default='covid19', choices=['covid19','mnist', 'cifar10'])
    parser.add_argument('-d', '--data_dir', type=str, default='/Users/jacklin/PycharmProjects/Final_Thesis/covid19-chest_CT_test', help='path to the dataset')
    parser.add_argument('-o', '--out_dir', type=str, default='/Users/jacklin/PycharmProjects/Final_Thesis/covid19-chest_CT_test', help='path where to save superpixels')
    parser.add_argument('-s', '--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of parallel threads')
    parser.add_argument('-n', '--n_sp', type=int, default=150, help='max number of superpixels per image')#复杂图片150
    parser.add_argument('-c', '--compactness', type=int, default=0.25, help='compactness of the SLIC algorithm '
                                                                      '(Balances color proximity and space proximity): '
                                                                      '0.25 is a good value for MNIST '
                                                                      'and 10 for color images like CIFAR-10')
    parser.add_argument('--seed', type=int, default=111, help='seed for shuffling nodes')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args
def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    print(nd,ty)
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
# type变为torch可处理的
def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()
def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def byte_to_torch():
    # process and save as torch files
    raw_folder = os.getcwd()
    processed_folder = raw_folder + '/torch_'
    if not os.path.isdir(processed_folder):
        os.makedirs(processed_folder)
    training_file, test_file = 'train.pt', 'test.pt'
    print('Processing...')
    training_set = (
        read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join(raw_folder, 'test-images-idx3-ubyte')),
        read_label_file(os.path.join(raw_folder, 'test-labels-idx1-ubyte'))
    )
    with open(os.path.join(processed_folder, training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(processed_folder, test_file), 'wb') as f:
        torch.save(test_set, f)

    print('Done of generating the torch data·················')


def get_torch_data(file_type):
    raw_folder = os.getcwd()
    processed_folder = raw_folder + '/torch_'
    if file_type == 'train':
        data_file = "train.pt"
    elif file_type == 'test':
        data_file = 'test.pt'
    else:
        data_file = ''
    data, targets = torch.load(os.path.join(processed_folder, data_file))
    return (data,targets)

def process_image(params):
    
    img, index, n_images, args, to_print, shuffle = params

    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)

    n_sp_extracted = args.n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    n_sp_query = args.n_sp + (20 if args.dataset == 'mnist' else 50)  # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    while n_sp_extracted > args.n_sp:
        superpixels = slic(img, n_segments=n_sp_query, compactness=args.compactness, multichannel=len(img.shape) > 2)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels

    assert n_sp_extracted <= args.n_sp and n_sp_extracted > 0, (args.split, index, n_sp_extracted, args.n_sp)
    assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))  # make sure superpixel indices are numbers from 0 to n-1

    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    if to_print:
        print('image={}/{}, shape={}, min={:.2f}, max={:.2f}, n_sp={}'.format(index + 1, n_images, img.shape,
                                                                              img.min(), img.max(), sp_intensity.shape[0]))

    return sp_intensity, sp_coord, sp_order, superpixels


if __name__ == '__main__':
    byte_to_torch()
    dt = datetime.datetime.now()
    print('start time:', dt)

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)  # to make node random permutation reproducible (not tested)

    # Read image data using torchvision
    args.split = 'train'
    if args.dataset == 'covid19':
        images, labels = get_torch_data(args.split)
        assert args.compactness < 10, ('high compactness can result in bad superpixels on MNIST')
        assert args.n_sp > 1 and args.n_sp < 64*64, ( #注意换成图片尺寸大小
            'the number of superpixels cannot exceed the total number of pixels or be too small')
    else:
        raise NotImplementedError('unsupported dataset: ' + args.dataset)


    if not isinstance(images, np.ndarray):
        images = images.numpy()
    if isinstance(labels, list):
        labels = np.array(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()

    n_images = len(labels)

    if args.threads <= 0:
        sp_data = []
        for i in range(n_images):
            sp_data.append(process_image((images[i], i, n_images, args, True, True)))
    else:
        with mp.Pool(processes=args.threads) as pool:
            sp_data  = pool.map(process_image, [(images[i], i, n_images, args, True, True) for i in range(n_images)])

    superpixels = [sp_data[i][3] for i in range(n_images)]
    sp_data = [sp_data[i][:3] for i in range(n_images)]
    with open('%s/%s_%dsp_%s.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump((labels.astype(np.int32), sp_data), f, protocol=2)
    with open('%s/%s_%dsp_%s_superpixels.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump(superpixels, f, protocol=2)

    print('done in {}'.format(datetime.datetime.now() - dt))


    args.split = 'test'
    if args.dataset == 'covid19':
        images, labels = get_torch_data(args.split)
        assert args.compactness < 10, ('high compactness can result in bad superpixels on MNIST')
        assert args.n_sp > 1 and args.n_sp < 64*64, ( #注意换成图片尺寸大小
            'the number of superpixels cannot exceed the total number of pixels or be too small')
    else:
        raise NotImplementedError('unsupported dataset: ' + args.dataset)


    if not isinstance(images, np.ndarray):
        images = images.numpy()
    if isinstance(labels, list):
        labels = np.array(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()

    n_images = len(labels)

    if args.threads <= 0:
        sp_data = []
        for i in range(n_images):
            sp_data.append(process_image((images[i], i, n_images, args, True, True)))
    else:
        with mp.Pool(processes=args.threads) as pool:
            sp_data  = pool.map(process_image, [(images[i], i, n_images, args, True, True) for i in range(n_images)])

    superpixels = [sp_data[i][3] for i in range(n_images)]
    sp_data = [sp_data[i][:3] for i in range(n_images)]
    with open('%s/%s_%dsp_%s.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump((labels.astype(np.int32), sp_data), f, protocol=2)
    with open('%s/%s_%dsp_%s_superpixels.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump(superpixels, f, protocol=2)

    print('done in {}'.format(datetime.datetime.now() - dt))