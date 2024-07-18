import random
import torch
import numpy as np
import nibabel as nib
from collections import OrderedDict


def load_mri_image(path):
    mri_image = nib.load(path)
    return mri_image.get_fdata().astype('float32'), mri_image.affine, mri_image.header


def get_psnr(image, reference, data_range=1., mask=None):
    squared_diff = (image - reference) ** 2
    if mask is not None:
        squared_diff = squared_diff * mask
        err = np.sum(squared_diff) / np.sum(mask)
    else:
        err = np.mean(squared_diff, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / err)


def get_rmse(x, y, mask=None):
    squared_diff = (x - y) ** 2

    if mask is not None:
        squared_diff = squared_diff * mask
        err = np.sum(squared_diff) / np.sum(mask)
    else:
        err = np.mean(squared_diff, dtype=np.float64)

    return np.sqrt(err)


def add_rician_noise(mri_image, sigma):
    noise_1 = np.random.normal(0, sigma, size=mri_image.shape).astype('float32')
    noise_2 = np.random.normal(0, sigma, size=mri_image.shape).astype('float32')

    noisy = (mri_image + noise_1) ** 2. + noise_2 ** 2.
    noisy = np.sqrt(noisy)

    return noisy, noisy.min(), noisy.max()


def split_dataset(files, test_ids_file):
    with open(test_ids_file, 'r') as f:
        test_ids = f.read().splitlines()

    train_files, test_files = [], []
    for file in files:
        tokens = file.split('-')
        if tokens[0] in test_ids:
            test_files.append(file)
        else:
            train_files.append(file)

    return train_files, test_files


def correct_model_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def mod_crop(img, mod):
    size = img.shape[:2]
    size = size - np.mod(size, mod)
    img = img[:size[0], :size[1], :]

    return img


def mod_pad(img, mod, mode='constant', channel_first=True):
    size = img.shape

    if img.ndim == 4 and channel_first:
        _, h, w, d = np.mod(size, mod)
    elif img.ndim == 4 and not channel_first:
        h, w, d, _ = np.mod(size, mod)
    else:
        h, w, d = np.mod(size, mod)

    h = mod - h if h else 0
    w = mod - w if w else 0
    d = mod - d if d else 0

    if w % 2 == 0:
        w_left = w_right = w // 2
    else:
        w_left, w_right = w // 2, w // 2 + 1

    if h % 2 == 0:
        h_up = h_down = h // 2
    else:
        h_up, h_down = h // 2, h // 2 + 1

    if d % 2 == 0:
        d_front = d_back = d // 2
    else:
        d_front, d_back = d // 2, d // 2 + 1

    if img.ndim == 4 and channel_first:
        pad = ((0, 0), (h_up, h_down), (w_left, w_right), (d_front, d_back))
    elif img.ndim == 4 and not channel_first:
        pad = ((h_up, h_down), (w_left, w_right), (d_front, d_back), (0, 0))
    else:
        pad = ((h_up, h_down), (w_left, w_right), (d_front, d_back))

    if mode == 'constant':
        img = np.pad(img, pad, mode=mode, constant_values=0)
    else:
        img = np.pad(img, pad, mode=mode)

    return img,  (h_up, h_down, w_left, w_right, d_front, d_back)    # size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_ensemble(image, axis=None):       # [batch, channels, height, width, depth]
    if axis is None:
        axis = (2, 3)

    ensemble = [
        image,
        torch.flip(image, (axis[0],)),
        torch.flip(image, (axis[1],)),
        torch.flip(image, axis)
    ]

    image_rot = torch.rot90(image, k=1, dims=axis)

    ensemble_rot = [
        image_rot,
        torch.flip(image_rot, (axis[0],)),
        torch.flip(image_rot, (axis[1],)),
        torch.flip(image_rot, axis)
    ]

    return ensemble + ensemble_rot


def split_ensemble(ensemble, axis=None):
    if axis is None:
        axis = (2, 3)

    # Apply inverse transforms to non-rotated images
    image = ensemble[0] + torch.flip(ensemble[1], (axis[0],)) + torch.flip(ensemble[2], (axis[1],)) + torch.flip(ensemble[3], axis)

    # First, apply inverse transform to non-rotated images anf then apply rotation in order to restore orientation.
    ensemble_rot = [
        ensemble[4],
        torch.flip(ensemble[5], (axis[0],)),
        torch.flip(ensemble[6], (axis[1],)),
        torch.flip(ensemble[7], axis)
    ]

    for e in ensemble_rot:
        image += torch.rot90(e, k=3, dims=axis)
    # Return denormalized image
    return image
