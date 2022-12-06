"""
Run detectron2 training for text block | heading | separator detection.

Available yaml-configs (prefix with "COCO-InstanceSegmentation/"):
+ mask_rcnn_R_50_C4_1x.yaml
+ mask_rcnn_R_50_C4_3x.yaml
+ mask_rcnn_R_50_DC5_1x.yaml
+ mask_rcnn_R_50_DC5_3x.yaml
+ mask_rcnn_R_50_FPN_1x.yaml
+ mask_rcnn_R_50_FPN_1x_giou.yaml
+ mask_rcnn_R_50_FPN_3x.yaml
+ mask_rcnn_R_101_C4_3x.yaml
+ mask_rcnn_R_101_DC5_3x.yaml
+ mask_rcnn_R_101_FPN_3x.yaml
R mask_rcnn_X_101_32x8d_FPN_3x.yaml

Available py-configs (prefix with "new_baselines/"):
- mask_rcnn_R_50_FPN_[50|100|200|400]ep_LSJ.py
- mask_rcnn_R_101_FPN_[100|200|400]ep_LSJ.py
- mask_rcnn_regnetx_4gf_dds_FPN_[100|200|400]ep_LSJ.py
- mask_rcnn_regnety_4gf_dds_FPN_[100|200|400]ep_LSJ.py
"""

import logging

import json
import os
import sys

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, launch, create_ddp_model, AMPTrainer, hooks, default_writers, \
    SimpleTrainer, default_setup
from detectron2 import model_zoo
from detectron2.evaluation import inference_on_dataset, print_csv_format, COCOEvaluator
from detectron2.model_zoo import get_config
from detectron2.config import get_cfg, LazyConfig, instantiate
import argparse

from detectron2.utils import comm

CLASSES = ["textblock", "heading"]


def load_data(data_dir, t="train"):
    if t == "train":
        with open(os.path.join(data_dir, "train.json"), 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open(os.path.join(data_dir, "val.json"), 'r') as file:
            val = json.load(file)
    return val


def custom_config_py(num_classes, output_dir, model="new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py", learning_rate=0.1):
    cfg = get_config(model)

    # DATASETS
    cfg.dataloader.test.dataset.names = ('val',)
    cfg.dataloader.train.dataset.names = ('train',)

    cfg.train.output_dir = output_dir

    cfg.optimizer.lr = learning_rate

    # cfg.train.device = 'cpu'

    # cfg.train.amp.enabled = False

    # cfg.model.backbone.bottom_up.norm = "BN"
    # cfg.model.backbone.norm = "BN"

    cfg.model.roi_heads.num_classes = num_classes

    cfg.dataloader.train.total_batch_size = 16

    cfg.train.eval_period = 1

    cfg.train.max_iter = 100000

    print(LazyConfig.to_py(cfg))

    return cfg


def custom_config_yaml(num_classes, output_dir, model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", learning_rate=0.1):
    cfg = get_cfg()
    cfg = get_config(model)

    # get configuration from model_zoo
    # cfg.merge_from_file(model_zoo.get_config_file(model))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # cfg.MODEL.RESNETS.DEPTH = 34
    # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    # Solver
    # cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = (20, 10000, 20000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.IMS_PER_BATCH = 16

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 30

    # INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    # DATASETS
    cfg.DATASETS.TEST = ('val',)
    cfg.DATASETS.TRAIN = ('train',)

    # DATASETS
    cfg.OUTPUT_DIR = output_dir

    print(cfg.dump())

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Mask R-CNN training for text block detection.")
    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--data_dir",
                        metavar="DIR",
                        help="path to the folder where the data is stored.")
    parser.add_argument("--output_dir",
                        metavar="DIR",
                        help="path to the output directory")
    parser.add_argument("--num_gpus",
                        type=int,
                        help="number of GPUs to run training on.")

    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument("--num_classes",
                        type=int,
                        help="number of GT classes")
    return parser


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
     ret = inference_on_dataset(
         model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
     )
     print_csv_format(ret)
     return ret


def main(args, num_classes, output_dir, config_file):
    for d in ["train", "val"]:
        DatasetCatalog.register(d, lambda d=d: load_data(args.data_dir, d))
        # MetadataCatalog.get(d).set(thing_classes=["textblock", "heading", "separator"])
        MetadataCatalog.get(d).set(thing_classes=CLASSES)
    metadata = MetadataCatalog.get("train")

    lr = 0.01
    if args.num_gpus:
        bs = (args.num_gpus * 2)
        lr = 0.02 * bs / 16

    if config_file.endswith(".py"):
        cfg = custom_config_py(num_classes=num_classes,
                               output_dir=output_dir,
                               model=config_file,
                               learning_rate=lr)

        model = instantiate(cfg.model)
        logger = logging.getLogger("detectron2")
        logger.info("Model:\n{}".format(model))
        model.to(cfg.train.device)

        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)

        train_loader = instantiate(cfg.dataloader.train)

        # cfg.dataloader.evaluator = COCOEvaluator('val', output_dir=output_dir)
        # default_setup(cfg, args)

        model = create_ddp_model(model, **cfg.train.ddp)
        trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
        checkpointer = DetectionCheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
        )
        trainer.register_hooks(
            [
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None,
                # hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None,
            ]
        )

        resume = True
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=resume)
        if resume and checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            start_iter = trainer.iter + 1
        else:
            start_iter = 0
        trainer.train(start_iter, cfg.train.max_iter)
    else:
        cfg = custom_config_yaml(num_classes=num_classes,
                                 output_dir=output_dir,
                                 model=config_file,
                                 learning_rate=lr)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        return trainer.train()


if __name__ == '__main__':
    args = get_parser().parse_args()

    # for d in ["train", "val"]:
    #     DatasetCatalog.register(d, lambda d=d: load_data(args.data_dir, d))
    #     MetadataCatalog.get(d).set(thing_classes=["textblock", "heading", "separator"])
    # metadata = MetadataCatalog.get("train")

    config_file = ""
    if args.config_file is not None:
        config_file = args.config_file
    else:
        config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        config_file = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        config_file = "new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py"

    output_dir = ""
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.data_dir, "output_" + config_file.split("/")[-1].split(".")[0])
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist, create it..")
        os.makedirs(output_dir)

    num_gpus = 0
    if args.num_gpus is not None:
        num_gpus = args.num_gpus

    num_classes = len(CLASSES)
    if args.num_classes is not None:
        num_classes = args.num_classes

    launch(main, num_gpus, dist_url=args.dist_url, args=(args, num_classes, output_dir, config_file,))
