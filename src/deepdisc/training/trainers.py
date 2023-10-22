from deepdisc.model import models
from deepdisc.model import loaders
from astrodet.astrodet import LazyAstroTrainer
from astrodet import detectron as detectron_addons
from detectron2.config import instantiate



def return_lazy_trainer(model,loader,optimizer,cfg,cfg_loader,hooklist):
    """Return a trainer for models built on LazyConfigs

    Parameters
    ----------
    model : torch model
        pointer to file
    loader : detectron2 data loader
    
    optimizer: detectron2 optimizer

    cfg : .py file
        The LazyConfig used to build the model

    cfg_loader: .yml file
        The config used for the data loaders
    
    hooklist: list
        The list of hooks to use for the trainer

    Returns
    -------
        trainer
    """
    trainer = LazyAstroTrainer(model, loader, optimizer, cfg, cfg_loader)
    trainer.register_hooks(hooklist)

    return trainer


def return_savehook(output_name):
    """Returns a hook for saving the model

    Parameters
    ----------
    output_name : str
        name of output file to save

    Returns
    -------
        a SaveHook
    """
    saveHook = detectron_addons.SaveHook()
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

def return_evallosshook(val_per,model,test_loader):
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