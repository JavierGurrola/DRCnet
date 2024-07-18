import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, Sampler
from skimage.util import view_as_windows
from utils import mod_pad, add_rician_noise


def load_mri_image(path, sequence='T1'):
    mri_image = nib.load(path).get_fdata().astype('float32')

    # Transpose image to use horizontal plane.
    # T1w
    if sequence == 'T1':
        mri_image = np.transpose(mri_image, (1, 0, 2))
    # T2w. PDw
    else:
        mri_image = np.transpose(mri_image, (2, 1, 0))
        mri_image = mri_image[:, ::-1, :].copy()

    return mri_image


def create_patches(image, patch_size, step):
    image = view_as_windows(image, patch_size, step)
    h_patches, w_patches, d_patches = image.shape[:3]                           # Number of patches per dimension.
    h, w, d = image.shape[3:]                                                   # Patch size.
    image = np.reshape(image, (h_patches * w_patches * d_patches, h, w, d))     # "list" of patches.

    return image


class DataSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        self.perm = []

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self._num_samples is not None:
            while len(self.perm) < self._num_samples:
                perm = np.random.permutation(n).astype('int32').tolist()
                self.perm.extend(perm)
            idx = self.perm[:self._num_samples]
            self.perm = self.perm[self._num_samples:]
        else:
            idx = np.random.permutation(n).astype('int32').tolist()

        return iter(idx)

    def __len__(self):
        return self.num_samples


class NoisyTrainingDataset(Dataset):
    def __init__(self, root_path, files, sequence='T1', patch_size=32, transform=None):
        self.root_path = root_path
        self.files = files
        self.sequence = sequence
        self.patch_size = patch_size
        self.transform = transform
        self.dataset = {'image': [], 'max val': []}
        self.load_dataset()

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        sample = {'image': self.dataset['image'][idx], 'max val': self.dataset['max val'][idx]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample['noisy'], sample['image'], sample['max val']

    def load_dataset(self):
        # print('Loading training dataset:')
        for i, file in enumerate(self.files, 1):
            mri_image = load_mri_image(os.path.join(self.root_path, file), self.sequence)
            shape = np.array(mri_image.shape)
            if np.any(shape < self.patch_size):
                mri_image, _ = mod_pad(mri_image, self.patch_size, mode='constant')

            mri_image = np.expand_dims(mri_image, 0)
            max_val = torch.tensor(mri_image.max(), dtype=torch.float32)
            mri_image = torch.from_numpy(mri_image)

            # print('{} / {} - file: {} - shape: {} - max val: {}'.format(
            #     i, len(self.files), file, mri_image.shape, max_val.item()))
            self.dataset['image'].append(mri_image)
            self.dataset['max val'].append(max_val)


class NoisyValidationDataset(Dataset):
    def __init__(self, root_path, files, noise_level, normalization_const, sequence='T1', patch_size=32, transform=None):
        self.root_path = root_path
        self.files = files
        self.noise_level = noise_level
        self.normalization_const = normalization_const
        self.sequence = sequence
        self.patch_size = patch_size
        self.transform = transform
        self.dataset = {'image': [], 'noisy': [], 'max val': []}
        self.load_dataset()

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        sample = {
            'image': self.dataset['image'][idx],
            'noisy': self.dataset['noisy'][idx],
            'max val': self.dataset['max val'][idx]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample['noisy'], sample['image'], sample['max val']

    def load_dataset(self):
        # print("Loading validation dataset:")
        patch_size = (self.patch_size, self.patch_size, self.patch_size)
        for image_idx, file in enumerate(self.files):
            clean_image = load_mri_image(os.path.join(self.root_path, file), self.sequence)

            max_val = clean_image.max()
            sigma = self.noise_level * max_val
            max_val = torch.tensor(max_val, dtype=torch.float32)

            clean_image, _ = mod_pad(clean_image, self.patch_size, mode='constant')
            noisy_image, _, _ = add_rician_noise(clean_image, sigma)

            # print('{} / {} - file: {} - shape: {} - max val: {}'.format(
            #     image_idx + 1, len(self.files), file, clean_image.shape, max_val))

            clean_patches = create_patches(clean_image, patch_size, patch_size)
            noisy_patches = create_patches(noisy_image, patch_size, patch_size)

            for clean, noisy in zip(clean_patches, noisy_patches):
                if np.any(clean > 0):
                    clean, noisy = np.expand_dims(clean, axis=0), np.expand_dims(noisy, axis=0)
                    clean = torch.from_numpy(clean.astype('float32')) / self.normalization_const
                    noisy = torch.from_numpy(noisy.astype('float32')) / self.normalization_const
                    self.dataset['image'].append(clean)
                    self.dataset['noisy'].append(noisy)
                    self.dataset['max val'].append(max_val)
