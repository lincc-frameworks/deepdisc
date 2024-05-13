import gc
import time

import detectron2.checkpoint as checkpointer
import torch
from detectron2.config import instantiate
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm

from deepdisc.astrodet import detectron as detectron_addons

class LazyAstroTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer)

        # Borrowed from DefaultTrainer constructor
        # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/defaults.html#DefaultTrainer
        self.checkpointer = checkpointer.DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
        )
        # load weights
        self.checkpointer.load(cfg.train.init_checkpoint)

        # record loss over iteration
        self.lossList = []
        self.vallossList = []

        self.period = 20
        self.iterCount = 0

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # self.scheduler = instantiate(cfg.lr_multiplier)
        self.valloss = 0

    # Note: print out loss over p iterations
    def set_period(self, p):
        self.period = p

    # Copied directly from SimpleTrainer, add in custom manipulation with the loss
    # see https://detectron2.readthedocs.io/en/latest/_modules/detectron2/engine/train_loop.html#SimpleTrainer
    def run_step(self):
        self.iterCount = self.iterCount + 1
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        # Note: in training mode, model() returns loss
        start = time.perf_counter()
        loss_dict = self.model(data)
        loss_time = time.perf_counter() - start

        # print('Loss dict',loss_dict)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
            all_losses = [l.cpu().detach().item() for l in loss_dict.values()]
        self.optimizer.zero_grad()
        losses.backward()

        # self._write_metrics(loss_dict,data_time)

        self.optimizer.step()

        self.lossList.append(losses.cpu().detach().numpy())
        if self.iterCount % self.period == 0 and comm.is_main_process():
            # print("Iteration: ", self.iterCount, " time: ", data_time," loss: ",losses.cpu().detach().numpy(), "val loss: ",self.valloss, "lr: ", self.scheduler.get_lr())
            print(
                "Iteration: ",
                self.iterCount,
                " data time: ",
                data_time,
                " loss time: ",
                loss_time,
                loss_dict.keys(),
                all_losses,
                "val loss: ",
                self.valloss,
                "lr: ",
                self.scheduler.get_lr(),
            )

        #del data
        #gc.collect()
        #torch.cuda.empty_cache()

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    def add_val_loss(self, val_loss):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        self.vallossList.append(val_loss)


def return_lazy_trainer(model, loader, optimizer, cfg, hooklist):
    """Return a trainer for models built on LazyConfigs

    Parameters
    ----------
    model : torch model
        pointer to file
    loader : detectron2 data loader

    optimizer : detectron2 optimizer

    cfg : .py file
        The LazyConfig used to build the model, and also stores config vals for data loaders

    hooklist : list
        The list of hooks to use for the trainer

    Returns
    -------
        trainer
    """
    trainer = LazyAstroTrainer(model, loader, optimizer, cfg)
    trainer.register_hooks(hooklist)
    return trainer


def return_savehook(output_name, save_period):
    """Returns a hook for saving the model

    Parameters
    ----------
    output_name : str
        name of output file to save

    Returns
    -------
        a SaveHook
    """
    #saveHook = detectron_addons.SaveHook()
    saveHook = detectron_addons.NewSaveHook(save_period)
    saveHook.set_output_name(output_name)
    return saveHook


def return_schedulerhook(optimizer):
    """Returns a hook for the learning rate

    Parameters
    ----------
    optimizer : detectron2 optimizer
        the optimizer that controls the learning rate

    Returns
    -------
        a CustomLRScheduler hook
    """
    schedulerHook = detectron_addons.CustomLRScheduler(optimizer=optimizer)
    return schedulerHook


def return_evallosshook(val_per, model, test_loader):
    """Returns a hook for evaulating the loss

    Parameters
    ----------
    val_per : int
        the frequency with which to calculate validation loss
    model: torch.nn.module
        the model
    test_loader: data loader
        the loader to read in the eval data

    Returns
    -------
        a LossEvalHook
    """
    lossHook = detectron_addons.LossEvalHook(val_per, model, test_loader)
    return lossHook


def return_optimizer(cfg):
    """Returns an optimizer for training

    Parameters
    ----------
    cfg : .py file
        The LazyConfig used to build the model

    Returns
    -------
        a pytorch optimizer
    """
    optimizer = instantiate(cfg.optimizer)
    return optimizer
