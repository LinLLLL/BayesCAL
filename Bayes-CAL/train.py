import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.ColoredMNIST
import datasets.ColoredCatsDogs
import datasets.PACS
import datasets.VLCS
import datasets.NICO
import datasets.NICO_BN
import datasets.NICO_SUB
import datasets.PACS_BN

import trainers.coop
import trainers.zsclip
import trainers.BCAL
import trainers.BCAL_LV
import trainers.BCAL_W2V
import trainers.BCAL_PL
import trainers.CAL
import trainers.CoCoOp
import trainers.DPLCLIP
import trainers.CONVENTION


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BCAL = CN()
    cfg.TRAINER.BCAL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.BCAL.CSC = False  # class-specific context
    cfg.TRAINER.BCAL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.BCAL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.BCAL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BCAL_PL = CN()
    cfg.TRAINER.BCAL_PL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.BCAL_PL.CSC = False  # class-specific context
    cfg.TRAINER.BCAL_PL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.BCAL_PL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.BCAL_PL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BCAL_LV = CN()
    cfg.TRAINER.BCAL_LV.N_CTX = 16  # number of context vectors
    cfg.TRAINER.BCAL_LV.CSC = False  # class-specific context
    cfg.TRAINER.BCAL_LV.CTX_INIT = ""  # initialization words
    cfg.TRAINER.BCAL_LV.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.BCAL_LV.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BCAL_W2V = CN()
    cfg.TRAINER.BCAL_W2V.N_CTX = 16  # number of context vectors
    cfg.TRAINER.BCAL_W2V.CSC = False  # class-specific context
    cfg.TRAINER.BCAL_W2V.CTX_INIT = ""  # initialization words
    cfg.TRAINER.BCAL_W2V.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.BCAL_W2V.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.CAL = CN()
    cfg.TRAINER.CAL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CAL.CSC = False  # class-specific context
    cfg.TRAINER.CAL.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CAL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CAL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.DPLCLIP = CN()
    cfg.TRAINER.DPLCLIP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.DPLCLIP.CSC = False  # class-specific context
    cfg.TRAINER.DPLCLIP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.DPLCLIP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.DPLCLIP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.CONVENTION = CN()
    cfg.TRAINER.CONVENTION.N_CTX = 16  # number of context vectors
    cfg.TRAINER.CONVENTION.CSC = False  # class-specific context
    cfg.TRAINER.CONVENTION.CTX_INIT = ""  # initialization words
    cfg.TRAINER.CONVENTION.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.CONVENTION.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
