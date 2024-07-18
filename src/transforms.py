import random
import torch
import numpy as np


class RandomFlip(object):
    def __init__(self, axis=1, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, noisy = sample['image'], sample.get('noisy', None)

            if isinstance(image, np.ndarray):
                image = np.flip(image, axis=self.axis)
                if noisy is not None:
                    noisy = np.flip(noisy, axis=self.axis)
            else:
                image = torch.flip(image, dims=[self.axis])
                if noisy is not None:
                    noisy = torch.flip(noisy, dims=[self.axis])

            sample['image'], sample['noisy'] = image, noisy

        return sample


class RandomRotation(object):
    def __init__(self, axes=(1, 2), p=0.5):
        self.axes = axes
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, noisy = sample['image'], sample.get('noisy', None)

            if isinstance(image, np.ndarray):
                image = np.rot90(image, axes=self.axes)
                if noisy is not None:
                    noisy = np.rot90(noisy, axes=self.axes)
            else:
                image = torch.rot90(image, dims=self.axes)
                if noisy is not None:
                    noisy = torch.rot90(noisy, dims=self.axes)

            sample['image'], sample['noisy'] = image, noisy

        return sample


class NoiseGenerator(object):
    def __init__(self, noise_level, fixed=True):
        self.noise_level = noise_level
        self.fixed = fixed

    def __call__(self, sample):
        image, max_val = sample.get('image'), sample.get('max val')

        if self.fixed:
            sigma = self.noise_level * max_val
        else:
            noise_level = int(100 * self.noise_level)
            noise_level = random.randint(1, noise_level)
            sigma = noise_level * max_val / 100.

        if isinstance(image, np.ndarray):
            noise_1 = np.random.normal(0, sigma, image.shape).astype('float32')
            noise_2 = np.random.normal(0, sigma, image.shape).astype('float32')
        else:
            noise_1 = torch.normal(mean=0, std=sigma, size=image.size(), device=image.device, dtype=torch.float32)
            noise_2 = torch.normal(mean=0, std=sigma, size=image.size(), device=image.device, dtype=torch.float32)

        noisy = (image + noise_1) ** 2 + noise_2 ** 2
        noisy = noisy ** 0.5
        sample['noisy'] = noisy

        return sample


class Normalize(object):
    def __init__(self, normalization_const):
        self.normalization_const = normalization_const

    def __call__(self, sample):
        image, noisy = sample['image'], sample['noisy']
        image, noisy = image / self.normalization_const, noisy / self.normalization_const
        sample['image'], sample['noisy'] = image, noisy
        return sample


class RandomPatches(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        patch_size = self.patch_size
        image, noisy = sample['image'], sample.get('noisy', None)
        _, h, w, d = image.size()

        while True:
            # Get crops points
            top = random.randint(0, h - patch_size)  # Inclusive rand
            bottom = top + patch_size

            left = random.randint(0, w - patch_size)
            right = left + patch_size

            front = random.randint(0, d - patch_size)
            back = front + patch_size

            image_patch = image[..., top: bottom, left: right, front: back]

            if noisy is not None:
                noisy_patch = noisy[..., top: bottom, left: right, front: back]
            else:
                noisy_patch = None

            if torch.any(image_patch > 0):
                break

        sample['image'], sample['noisy'] = image_patch, noisy_patch

        return sample
