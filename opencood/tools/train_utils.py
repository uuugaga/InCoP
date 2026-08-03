# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
from contextlib import contextmanager
import torch
import torch.optim as optim

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def check_missing_key(model_state_dict, ckpt_state_dict):
    checkpoint_keys = set(ckpt_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    missing_keys = model_keys - checkpoint_keys
    extra_keys = checkpoint_keys - model_keys

    missing_key_modules = set([keyname.split('.')[0] for keyname in missing_keys])
    extra_key_modules = set([keyname.split('.')[0] for keyname in extra_keys])

    print("------ Loading Checkpoint ------")
    if len(missing_key_modules) == 0 and len(extra_key_modules) ==0:
        return

    print("Missing keys from ckpt:")
    print(*missing_key_modules,sep='\n',end='\n\n')
    # print(*missing_keys,sep='\n',end='\n\n')

    print("Extra keys from ckpt:")
    print(*extra_key_modules,sep='\n',end='\n\n')
    print(*extra_keys,sep='\n',end='\n\n')

    print("You can go to tools/train_utils.py to print the full missing key name!")
    print("--------------------------------")


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


class EpochWarmupScheduler:
    """Epoch-level LR warmup wrapper for schedulers stepped after each epoch."""

    def __init__(self, optimizer, scheduler, warmup_epochs=0, start_factor=0.1, init_epoch=0):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = int(max(warmup_epochs, 0))
        self.start_factor = float(start_factor)
        if not 0.0 < self.start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1].")
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._set_lr_for_epoch(int(max(init_epoch, 0)))

    def _warmup_factor(self, epoch):
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return 1.0
        progress = float(epoch) / float(max(self.warmup_epochs, 1))
        return self.start_factor + (1.0 - self.start_factor) * progress

    def _set_lr_for_epoch(self, epoch):
        factor = self._warmup_factor(epoch)
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor

    def step(self, epoch=None):
        next_epoch = 0 if epoch is None else int(epoch) + 1
        if next_epoch < self.warmup_epochs:
            self._set_lr_for_epoch(next_epoch)
            return
        self.scheduler.step(next_epoch)


def setup_lr_schedular(hypes, optimizer, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    warmup_epochs = int(lr_schedule_config.get("warmup_epochs", 0))
    if warmup_epochs > 0:
        start_factor = float(lr_schedule_config.get("warmup_start_factor", 0.1))
        return EpochWarmupScheduler(
            optimizer,
            scheduler,
            warmup_epochs=warmup_epochs,
            start_factor=start_factor,
            init_epoch=last_epoch,
        )

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler



class ModelEMA:
    """Exponential moving average of model state for smoother evaluation."""

    def __init__(self, model, decay=0.999, warmup_steps=0, use_for_eval=True):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema.decay must be in [0, 1).")
        self.decay = float(decay)
        self.warmup_steps = int(max(warmup_steps, 0))
        self.use_for_eval = bool(use_for_eval)
        self.num_updates = 0
        self.shadow = {}
        self._copy_from(model)

    def _copy_from(self, model):
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_tensor(value)
        }

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        model_state = model.state_dict()
        if self.num_updates <= self.warmup_steps:
            self._copy_from(model)
            return
        for key, value in model_state.items():
            if not torch.is_tensor(value):
                continue
            value = value.detach()
            if key not in self.shadow:
                self.shadow[key] = value.clone()
            elif torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self):
        return {key: value.detach().cpu() for key, value in self.shadow.items()}

    @contextmanager
    def average_parameters(self, model):
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_tensor(value)
        }
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=False)


def setup_model_ema(hypes, model, steps_per_epoch):
    ema_cfg = hypes.get("ema", {}) or {}
    if isinstance(ema_cfg, bool):
        ema_cfg = {"enabled": ema_cfg}
    if not ema_cfg.get("enabled", False):
        return None
    decay = float(ema_cfg.get("decay", 0.999))
    warmup_steps = int(ema_cfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_epochs = float(ema_cfg.get("warmup_epochs", 0.0))
        warmup_steps = int(round(max(warmup_epochs, 0.0) * max(int(steps_per_epoch), 1)))
    use_for_eval = bool(ema_cfg.get("use_for_eval", True))
    print(
        f"EMA enabled: decay={decay}, warmup_steps={warmup_steps}, "
        f"use_for_eval={use_for_eval}"
    )
    return ModelEMA(
        model,
        decay=decay,
        warmup_steps=warmup_steps,
        use_for_eval=use_for_eval,
    )

def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)
