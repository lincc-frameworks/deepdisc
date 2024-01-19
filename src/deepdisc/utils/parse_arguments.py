import argparse
import os
import sys

import numpy as np


def make_inference_arg_parser():
    """Create the parser for DeepDisc inference, including common arguments used by
    detectron2 users.

    Returns
    -------
    parser : ArgumentParser
        The argument parser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--datatype", default=8, type=int)
    parser.add_argument("--nc", default=2, type=int)
    parser.add_argument("--norm", default="raw", type=str, help="contrast scaling")
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument("--roi-thresh", default=0.1, type=float)
    parser.add_argument("--run-name", default="Swin_test.pth", type=str)
    parser.add_argument("--savedir", default=".", type=str)
    parser.add_argument("--scheme", default=2, type=int, help="classification scheme")
    parser.add_argument("--testfile", default="/home/shared/hsc/HSC/HSC_DR3/data/single_test.json", type=str)
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    
    # To differentiate the kind of run 
    parser.add_argument("--use-dc2", default=False, action="store_true")
    parser.add_argument("--use-redshift", default=False, action="store_true")
    
    return parser


def make_training_arg_parser(epilog=None):
    """Create the parser for DeepDisc training, including common arguments used by
    detectron2 users.

    Parameters
    ----------
    epilog: str
        The epilog passed to ArgumentParser describing the usage.

    Returns
    -------
    parser : ArgumentParser
        The argument parser.
    """
    # Create the initial parser.
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add arguments for the run.
    run_args = parser.add_argument_group("Basic run arguments")
    run_args.add_argument(
        "--cfgfile",
        type=str,
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        help="path to model config file",
    )
    run_args.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    run_args.add_argument(
        "--data-dir",
        type=str,
        default="/home/shared/hsc/HSC/HSC_DR3/data/",
        help="directory with data",
    )

    run_args.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    run_args.add_argument(
        "--from-scratch",
        action="store_true",
        help="use this if you don't want to use pretrained weights",
    )
    run_args.add_argument("--output-dir", type=str, default="./", help="output directory to save model")
    run_args.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    run_args.add_argument("--run-name", type=str, default="Swin_test", help="output name for run")
    
    # To differentiate the kind of run 
    run_args.add_argument("--use-dc2", default=False, action="store_true")
    run_args.add_argument("--use-redshift", default=False, action="store_true")

    # Add arguments for the machine specifications
    machine_args = parser.add_argument_group("Machine arguments")
    machine_args.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    machine_args.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    machine_args.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    # Add arguments for the data normalization and modeling.
    model_args = parser.add_argument_group("Model configuration arguments")
    model_args.add_argument("--A", type=float, default=1e3, help="scaling factor for int16")
    model_args.add_argument("--alphas", type=float, nargs="*", help="weights for focal loss")
    model_args.add_argument(
        "--cp",
        type=float,
        default=99.99,
        help="ceiling percentile for saturation cutoff",
    )
    model_args.add_argument("--do-fl", action="store_true", help="use focal loss")
    model_args.add_argument(
        "--do-norm",
        action="store_true",
        help="normalize input image (ignore if lupton)",
    )
    model_args.add_argument("--dtype", type=int, default=8, help="data type of array")
    model_args.add_argument("--modname", type=str, default="swin", help="")
    model_args.add_argument("--norm", type=str, default="raw", help="contrast scaling")
    model_args.add_argument("--Q", type=float, default=10, help="lupton Q")
    model_args.add_argument("--scheme", type=int, default=1, help="classification scheme")
    model_args.add_argument("--stretch", type=float, default=0.5, help="lupton stretch")
    model_args.add_argument("--tl", type=int, default=1, help="total size of training set")

    # Add a section of advanced arguments.
    adv_args = parser.add_argument_group("Advanced arguments")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    adv_args.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    adv_args.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def dtype_from_args(dt=32):
    """Returns the dtype corresponding to the dtype argument string.

    Parameters
    ----------
    dt: int
        The integer representing the number of bytes to use.
        8 = uint8
        16 = int16
        32 = float32 (default)

    Returns
    -------
    type
        The dtype to use.
    """
    if dt == 32:
        return np.float32
    elif dt == 16:
        return np.int16
    elif dt == 8:
        return np.uint8
    else:
        raise ValueError("Unknown dtype argument {dt}.")
