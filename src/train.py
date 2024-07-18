import csv
import os
import torch
import time
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from metrics import PSNR, SSIM


class EpochLogger:
    def __init__(self):
        self.log = {'train loss': 0., 'train psnr': 0., 'train ssim': 0., 'val loss': 0., 'val psnr': 0., 'val ssim': 0.}

    def update_log(self, metrics, phase):
        for key, value in metrics.items():
            self.log[''.join([phase, ' ', key])] += value

    def get_log(self, num_samples, phase):
        log = {
            phase + ' loss': self.log[phase + ' loss'] / num_samples,
            phase + ' psnr': self.log[phase + ' psnr'] / num_samples,
            phase + ' ssim': self.log[phase + ' ssim'] / num_samples
        }
        return log


class FileLogger(object):
    def __init__(self, file_path, header):
        self.file_path = file_path
        with open(self.file_path, 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(header)

    def __call__(self, log):
        # Format log file:
        log[1] = '{:.5e}'.format(log[1])    # Learning rate

        # Train loss, PSNR, SSIM
        log[2], log[3], log[4] = '{:.5e}'.format(log[2]), '{:.5f}'.format(log[3]), '{:.5f}'.format(log[4])

        # Val loss, PSNR, SSIM
        log[5], log[6], log[7] = '{:.5e}'.format(log[5]), '{:.5f}'.format(log[6]), '{:.5f}'.format(log[7])

        with open(self.file_path, 'a') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(log)


def fit_model(model, data_loaders, criterion, optimizer, scheduler, device, num_epochs, val_frequency,
              checkpoint_dir, model_name, normalization_const=255., verbose=True, clip_value=None):
    """
    Training of brain denoiser model.
    :param model:                   Model to fit.
    :param data_loaders:            Iterable data loaders with train and validation datasets.
    :param criterion:               Loss function.
    :param optimizer:               Optimizer algorithm.
    :param scheduler:               Learning rate scheduler.
    :param device:                  Device used during train (CPU/GPU).
    :param num_epochs:              Number of training epochs.
    :param val_frequency:           Number of epochs between validation stages
    :param checkpoint_dir:          Path to create and store training log file and model checkpoints.
    :param model_name:              Base name of the model saved in checkpoint_dir.
    :param normalization_const:     Constant used to rescale values to a range near to [0,1].
    :param verbose:                 If true, displays progress bar.
    :param clip_value:              Gradient's clip value.
    :return:
    """
    psnr = PSNR(data_range=1., reduction='none', eps=0)
    ssim = SSIM(1, data_range=1., size_average=False)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logfile_path = os.path.join(checkpoint_dir,  ''.join([model_name, '_logfile.csv']))
    model_path = os.path.join(checkpoint_dir, ''.join([model_name, '-{:03d}-{:.4e}-{:.4f}-{:.4f}.pth']))
    header = ['epoch', 'lr', 'train loss', 'train psnr', 'train ssim', 'val loss', 'val psnr', 'val ssim']
    file_logger = FileLogger(logfile_path, header)
    best_model_path, best_psnr = '', -np.inf
    since = time.time()

    for epoch in range(1, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        epoch_logger, epoch_log = EpochLogger(), dict()

        for phase in ['train', 'val']:
            num_samples = 0
            if phase == 'train':
                model.train()
            elif phase == 'val' and epoch % val_frequency == 0:
                model.eval()
            else:
                break

            if verbose:
                if phase == 'train':
                    print('\nEpoch: {}/{} - Learning rate: {:.4e}'.format(epoch, num_epochs, lr))
                    description = 'Training - Loss:{:.5e} - PSNR:{:.5f} - SSIM:{:.5f}'
                else:
                    description = 'Validation - Loss:{:.5e} - PSNR:{:.5f} - SSIM:{:.5f}'
                iterator = tqdm(enumerate(data_loaders[phase], 1), total=len(data_loaders[phase]), ncols=120)
                iterator.set_description(description.format(0, 0, 0))
            else:
                iterator = enumerate(data_loaders[phase], 1)

            for step, (inputs, targets, max_val) in iterator:
                inputs, targets, max_val = inputs.to(device), targets.to(device), max_val.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    if clip_value is not None:
                        clip_grad_norm_(model.parameters(), max_norm=clip_value, norm_type=2)
                    optimizer.step()
                    if isinstance(scheduler, optim.lr_scheduler.CyclicLR):
                        scheduler.step()

                num_samples += inputs.size()[0]
                max_val = max_val.view(max_val.size()[0], 1, 1, 1, 1) / normalization_const
                outputs, targets = outputs / max_val, targets / max_val

                metrics = {
                    'loss': loss.item() * inputs.size()[0],
                    'psnr': psnr(outputs, targets).sum().item(),
                    'ssim': ssim(outputs, targets).sum().item()
                }
                epoch_logger.update_log(metrics, phase)
                log = epoch_logger.get_log(num_samples, phase)

                if verbose:
                    iterator.set_description(description.format(log[phase + ' loss'], log[phase + ' psnr'], log[phase + ' ssim']))

            if phase == 'val':
                # Use this scheduler in the validation dataset.
                if (scheduler is not None) and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(log['val psnr'])

                # Save the current best model with respect to the validation dataset.
                if log['val psnr'] > best_psnr:
                    best_psnr = log['val psnr']
                    best_model_path = model_path.format(epoch, log['val loss'], log['val psnr'], log['val ssim'])
                    torch.save(model.state_dict(), best_model_path)

            # Other schedulers (Except the cyclic) are evaluated in every epoch.
            elif (scheduler is not None) and (not isinstance(scheduler, optim.lr_scheduler.CyclicLR)):
                scheduler.step()

            epoch_log = {**epoch_log, **log}

        epoch_data = [epoch, lr, epoch_log['train loss'], epoch_log['train psnr'], epoch_log['train ssim'],
                      epoch_log.get('val loss', 0), epoch_log.get('val psnr', 0), epoch_log.get('val ssim', 0)]

        file_logger(epoch_data)

    best_model_path = model_path.format(epoch, log['val loss'], log['val psnr'], log['val ssim'])
    torch.save(model.state_dict(), best_model_path)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best PSNR: {:4f}'.format(best_psnr))
