import os
import sys
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
from torchvision import transforms

from model import DenoiserNet
from train import fit_model
from data_management import NoisyTrainingDataset, NoisyValidationDataset, DataSampler
from transforms import *
from utils import set_seed, split_dataset


if __name__ == '__main__':
    set_seed(1)
    with open('./config.yaml', 'r') as stream:                # Load YAML all configuration file.
        config = yaml.safe_load(stream)

    model_parameters = config['model']                      # Load parameters independently.
    training_params = config['train']
    valid_params = config['val']

    train_noise_level = training_params['noise level']      # Training parameters.
    epochs = training_params['epochs']
    dataset_splits = training_params['dataset splits']
    train_patch_size = training_params['patch size']
    train_batch_size = training_params['batch size']
    learning_rate = training_params['learning rate']
    weight_decay = training_params['weight decay']
    grad_clip = training_params['grad clip']
    scheduler_gamma = training_params['scheduler gamma']
    scheduler_step = training_params['scheduler step']

    dataset_path = training_params['dataset path']          # Local or cluster parameters
    workers = training_params['workers']
    checkpoint_dir = training_params['checkpoint path']
    verbose = training_params['verbose']
    device = training_params['device']
    multi_pgu = training_params['multi gpu']

    val_noise_level = valid_params['noise level']           # Validation parameters
    val_patch_size = valid_params['patch size']
    val_batch_size = valid_params['batch size']
    val_frequency = valid_params['frequency']

    sequence = model_parameters['sequence']
    normalization_const = model_parameters['normalization constant']
    dataset_path += sequence
    checkpoint_dir += ('-' + sequence)

    # Load and split dataset
    data_files = os.listdir(dataset_path)

    # Stratify split according to different MRI systems:
    #   Hammersmith Hospital using a Philips 3T: HH
    #   Guy’s Hospital using a Philips 1.5T: Guys

    hh_files = [file for file in data_files if 'HH' in file]
    guys_files = [file for file in data_files if 'Guys' in file]

    # Test files consider 10% of dataset
    train_hh_files, _ = split_dataset(hh_files, '../file_lists/hh_files.txt')
    train_guys_files, _ = split_dataset(guys_files, '../file_lists/guys_files.txt')

    # Split according to 0.8, 0.1 and 0.1 proportions with respect to the whole dataset
    train_split = 8. / 9.
    random.shuffle(train_hh_files)
    random.shuffle(train_guys_files)

    val_hh_files = train_hh_files[int(train_split * len(train_hh_files)):]
    val_guys_files = train_guys_files[int(train_split * len(train_guys_files)):]

    train_hh_files = train_hh_files[:int(train_split * len(train_hh_files))]
    train_guys_files = train_guys_files[:int(train_split * len(train_guys_files))]

    train_files = train_hh_files + train_guys_files
    val_files = val_hh_files + val_guys_files

    model_parameters = config['model']
    model = DenoiserNet(**model_parameters)
    model_name = 'model_braind-' + sequence
    param_group = []

    for name, param in model.named_parameters():
        if 'act' in name or 'bn' in name:
            p = {'params': param, 'weight_decay': 0.}
        else:
            p = {'params': param, 'weight_decay': weight_decay}
        param_group.append(p)

    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (model_parameters['input channels'], 64, 64, 64))

    print('Model summary:')
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    device = torch.device(device)
    print("Using device: {}".format(device))
    if torch.cuda.device_count() > 1 and 'cuda' in device.type and multi_pgu:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')

    model.to(device)

    training_transform = transforms.Compose([
        RandomPatches(train_patch_size),
        RandomFlip(1, p=0.5),
        RandomFlip(2, p=0.5),
        RandomFlip(3, p=0.5),
        RandomRotation((1, 2), p=0.5),
        RandomRotation((1, 3), p=0.5),
        RandomRotation((2, 3), p=0.5),
        NoiseGenerator(train_noise_level, fixed=False),
        Normalize(normalization_const=normalization_const)
    ])

    # train_files, val_files = train_files[:32], val_files[:10]

    training_dataset = NoisyTrainingDataset(dataset_path, train_files, sequence, train_patch_size, training_transform)
    validation_dataset = NoisyValidationDataset(dataset_path, val_files, val_noise_level,
                                                normalization_const, sequence, val_patch_size)

    # Update training parameters according to splits
    epochs = epochs * dataset_splits
    samples_per_epoch = 256 * len(training_dataset) // dataset_splits
    scheduler_step = scheduler_step * dataset_splits

    # Training in sub-epochs:
    print('Training samples:', len(training_dataset), '\nValidation samples:', len(validation_dataset))
    sampler = DataSampler(training_dataset, num_samples=samples_per_epoch)

    data_loaders = {
        'train': DataLoader(training_dataset, batch_size=train_batch_size, num_workers=workers, sampler=sampler),
        'val': DataLoader(validation_dataset, batch_size=val_batch_size, num_workers=workers),
    }

    # Optimization:
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(param_group, lr=learning_rate)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Fit the model
    fit_model(model, data_loaders, criterion, optimizer, lr_scheduler, device, epochs,
              val_frequency, checkpoint_dir, model_name, normalization_const, verbose, grad_clip)
