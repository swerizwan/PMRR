from __future__ import division
import os
import torch
import datetime
import logging
logger = logging.getLogger(__name__)

class CheckpointSaver():
    """Class that handles saving and loading checkpoints during training."""

    def __init__(self, save_dir, save_steps=1000, overwrite=False):
        """
        Initialize CheckpointSaver object.

        Args:
            save_dir (str): Directory where checkpoints will be saved.
            save_steps (int): Interval (in steps) at which to save checkpoints.
            overwrite (bool): Whether to overwrite the latest checkpoint or not.
        """
        self.save_dir = os.path.abspath(save_dir)
        self.save_steps = save_steps
        self.overwrite = overwrite
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.get_latest_checkpoint()
        return

    def exists_checkpoint(self, checkpoint_file=None):
        """
        Check if a checkpoint exists in the current directory.

        Args:
            checkpoint_file (str or None): Specific checkpoint file to check.

        Returns:
            bool: True if checkpoint exists, False otherwise.
        """
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)
    
    def save_checkpoint(self, models, optimizers, epoch, batch_idx, batch_size,
                        total_step_count, is_best=False, save_by_step=False, interval=5):
        """
        Save a checkpoint.

        Args:
            models (dict): Dictionary of models to save.
            optimizers (dict): Dictionary of optimizers to save.
            epoch (int): Current epoch number.
            batch_idx (int): Current batch index.
            batch_size (int): Size of batches.
            total_step_count (int): Total count of steps.
            is_best (bool): Flag to indicate if this is the best model so far.
            save_by_step (bool): Flag to save checkpoint by step count.
            interval (int): Interval to save checkpoint by epoch.

        """
        timestamp = datetime.datetime.now()
        if self.overwrite:
            checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, 'model_latest.pt'))
        elif save_by_step:
            checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, '{:08d}.pt'.format(total_step_count)))
        else:
            if epoch % interval == 0:
                checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, f'model_epoch_{epoch:02d}.pt'))
            else:
                checkpoint_filename = None
        
        checkpoint = {}
        for model in models:
            model_dict = models[model].state_dict()
            for k in list(model_dict.keys()):
                if k.startswith('iuv2smpl.smpl.'):
                    del model_dict[k]
            checkpoint[model] = model_dict
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        checkpoint['total_step_count'] = total_step_count
        print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
        
        if checkpoint_filename is not None:
            torch.save(checkpoint, checkpoint_filename)
            print('Saving checkpoint file [' + checkpoint_filename + ']')
        if is_best:
            checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, 'model_best.pt'))
            torch.save(checkpoint, checkpoint_filename)
            print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
            print('Saving checkpoint file [' + checkpoint_filename + ']')

    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        """
        Load a checkpoint.

        Args:
            models (dict): Dictionary of models to load checkpoint into.
            optimizers (dict): Dictionary of optimizers to load checkpoint into.
            checkpoint_file (str or None): Specific checkpoint file to load.

        Returns:
            dict: Dictionary containing epoch, batch_idx, batch_size, and total_step_count.
        """
        if checkpoint_file is None:
            logger.info('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        checkpoint = torch.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                model_dict = models[model].state_dict()
                pretrained_dict = {k: v for k, v in checkpoint[model].items()
                                   if k in model_dict.keys()}
                model_dict.update(pretrained_dict)
                models[model].load_state_dict(model_dict)
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        return {
            'epoch': checkpoint['epoch'],
            'batch_idx': checkpoint['batch_idx'],
            'batch_size': checkpoint['batch_size'],
            'total_step_count': checkpoint['total_step_count']
        }

    def get_latest_checkpoint(self):
        """
        Get filename of latest checkpoint if it exists.
        """
        checkpoint_list = [] 
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        # sort
        import re
        def atof(text):
            try:
                retval = float(text)
            except ValueError:
                retval = text
            return retval

        def natural_keys(text):
            """
            Sort filenames in natural (human) order.
            """
            return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
        
        checkpoint_list.sort(key=natural_keys)
        self.latest_checkpoint = None if len(checkpoint_list) == 0 else checkpoint_list[-1]
        return
