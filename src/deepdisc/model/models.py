from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model


def return_lazy_model(cfg):
    """Return a model formed from a LazyConfig.

    Parameters
    ----------
    cfg : .py file
        a LazyConfig

    Returns
    -------
        torch model
    """

    model = instantiate(cfg.model)

    model.to(cfg.train.device)
    model = create_ddp_model(model, **cfg.train.ddp)

    return model
