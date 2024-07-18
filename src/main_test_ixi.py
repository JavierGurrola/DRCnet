import yaml
import torch
import nibabel as nib
import numpy as np
import os
import time
from skimage.metrics import structural_similarity

from os.path import join
from model import DenoiserNet

from utils import correct_model_dict, mod_pad, add_rician_noise, load_mri_image, get_rmse, get_psnr


def predict(model, sequence, noise_level, noisy_dataset, clean_dataset, affines, max_vals, device, results_path,
            save_images=None):

    noisy_psnr_list, noisy_rmse_list, noisy_ssim_list = [], [], []
    psnr_list, rmse_list, ssim_list, time_list = [], [], [], []

    for i, (noisy, clean, affine, max_val) in \
            enumerate(zip(noisy_dataset, clean_dataset, affines, max_vals), 1):

        psnr_noisy = get_psnr(noisy, clean, max_val, mask=None)
        rmse_noisy = get_rmse(noise_level * normalization_const, clean * normalization_const)
        ssim_noisy = structural_similarity(noisy, clean, data_range=max_val, multichannel=False,
                                           gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

        noisy_psnr_list.append(psnr_noisy)
        noisy_rmse_list.append(rmse_noisy)
        noisy_ssim_list.append(ssim_noisy)

        noisy, size = mod_pad(noisy, 2, mode='reflect')
        noisy = np.reshape(noisy, (1, 1) + noisy.shape)  # Expand for channel and for batch.
        noisy = torch.from_numpy(noisy).to(device)

        with torch.no_grad():
            model.eval()
            start = time.time()  # Measure only estimation time, without considering transference
            estimated_image = model(noisy)
            end = time.time()
            estimated_image = estimated_image.detach().cpu().numpy().squeeze()

            if size[1] > 0:
                estimated_image = estimated_image[size[0]:-size[1], ...]

            if size[3] > 0:
                estimated_image = estimated_image[:, size[2]:-size[3], :]

            if size[5] > 0:
                estimated_image = estimated_image[..., size[4]:-size[5]]

        estimated_image = np.clip(estimated_image, 0, estimated_image.max()).copy()
        psnr = get_psnr(estimated_image, clean, max_val, mask=None)
        rmse = get_rmse(estimated_image * normalization_const, clean * normalization_const)
        ssim = structural_similarity(estimated_image, clean, data_range=max_val, multichannel=False,
                                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

        elapsed_time = end - start
        psnr_list.append(psnr)
        rmse_list.append(rmse)
        ssim_list.append(ssim)
        time_list.append(elapsed_time)

        message = 'Image {}: N. PSNR: {:.4f} - N. SSIM: {:.4f} - N. RMSE: {:.4f}' \
                  ' - PSNR: {:.4f} - SSIM: {:.4f} - RMSE: {:.4f} - time: {:.4e}'
        print(message.format(i, psnr_noisy, ssim_noisy, rmse_noisy, psnr, ssim, rmse, elapsed_time))

        if save_images is not None:
            # restore to their original arrangement
            if sequence == 'T1':
                estimated_image = np.transpose(estimated_image, (1, 0, 2))
            else:
                estimated_image = estimated_image[:, ::-1, :]
                estimated_image = np.transpose(estimated_image, (2, 1, 0))

            # Round values and save in 'uint16' to save disk space
            estimated_image = np.around(normalization_const * estimated_image)
            estimated_image = estimated_image.astype('uint16')

            new_image_path = str(noise_level) + '_' + save_images[i - 1]
            mri_image = nib.Nifti1Image(estimated_image * normalization_const, affine)
            nib.save(mri_image, os.path.join(results_path, new_image_path))

    return psnr_list, ssim_list, rmse_list, time_list, noisy_psnr_list, noisy_ssim_list, noisy_rmse_list


if __name__ == '__main__':
    np.random.seed(123)
    with open('./config.yaml', 'r') as stream:  # Load YAML all configuration file.
        config = yaml.safe_load(stream)

    model_parameters = config['model']
    test_parameters = config['test']

    noise_levels = test_parameters['noise levels']
    dataset_path = test_parameters['ixi dataset path']
    results_path = test_parameters['ixi results path']
    pretrained_models_path = test_parameters['pretrained models path']
    save_images = test_parameters['save images']
    device_name = test_parameters['device']

    sequence = model_parameters['sequence']
    normalization_const = model_parameters['normalization constant']
    model_name = "model_braind-{}.pth".format(sequence)

    model_path = join(pretrained_models_path, model_name)
    model = DenoiserNet(**model_parameters)
    device = torch.device(device_name)
    print("Using device: {}".format(device))

    state_dict = torch.load(model_path)
    state_dict = correct_model_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    dataset_path = dataset_path + sequence
    results_path = results_path + sequence

    with open('../file_lists/guys_files.txt') as f_guys, open('../file_lists/hh_files.txt') as f_hh:
        # test_ids = f_guys.read().splitlines() + f_hh.read().splitlines()
        # test_ids = f_hh.read().splitlines()
        test_ids = f_guys.read().splitlines()

        test_paths, save_paths = [], []
        all_files = os.listdir(dataset_path)

        for file in all_files:
            if any(test_id in file for test_id in test_ids):
                test_paths.append(join(dataset_path, file))
                save_paths.append(file)

    if not save_images:
        save_paths = None
    else:
        os.makedirs(results_path, exist_ok=True)

    for noise_level in noise_levels:
        noisy_dataset, clean_dataset, affines, max_vals = [], [], [], []
        print('Noise level: {}'.format(noise_level))

        # for test_path in tqdm(test_paths, desc='Loading images'):
        for test_path in test_paths:
            # Load MRI images
            data, affine, header = load_mri_image(test_path)
            if sequence == 'T1':
                data = np.transpose(data, (1, 0, 2))
            else:
                data = np.transpose(data, (2, 1, 0))
                data = data[:, ::-1, :].copy()
            affines.append(affine)

            # Add noise
            max_val = data.max()
            sigma = max_val * noise_level
            noisy, _, _ = add_rician_noise(data, sigma)
            max_vals.append(max_val / normalization_const)
            noisy, data = noisy / normalization_const, data / normalization_const

            noisy_dataset.append(noisy)
            clean_dataset.append(data)

        psnr, ssim, rmse, time_, noisy_psnr, noisy_ssim, noisy_rmse = \
            predict(model, sequence, noise_level, noisy_dataset, clean_dataset,
                    affines, max_vals, device, results_path, save_paths)

        psnr, ssim, rmse, time_, = np.mean(psnr), np.mean(ssim), np.mean(rmse), np.mean(time_)
        noisy_psnr, noisy_ssim, noisy_rmse = np.mean(noisy_psnr), np.mean(noisy_ssim), np.mean(noisy_rmse)
        psnr, ssim = np.around(psnr, decimals=4), np.around(ssim, decimals=4)

        message = 'PSNR: {:.4f} - SSIM: {:.4f} - RMSE: {:.4f} - time: {:.4e}' \
                  ' - Noisy PSNR: {:.4f} - Noisy SSIM: {:.4f} - Noisy RMSE: {:.4f}'
        print(message.format(psnr, ssim, rmse, time_, noisy_psnr, noisy_ssim, noisy_rmse))
   