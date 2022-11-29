import logging

import json
import os
import sys

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, launch, create_ddp_model, AMPTrainer, hooks, default_writers, \
    SimpleTrainer, default_setup, DefaultPredictor
from detectron2 import model_zoo
from detectron2.evaluation import inference_on_dataset, print_csv_format, COCOEvaluator
from detectron2.model_zoo import get_config
from detectron2.config import get_cfg, LazyConfig, instantiate
import argparse

from detectron2.utils import comm


def load_data(data_dir, t="train"):
    if t == "train":
        with open(os.path.join(data_dir, "train.json"), 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open(os.path.join(data_dir, "val.json"), 'r') as file:
            val = json.load(file)
    return val


def custom_config_py(num_classes, output_dir, model_weights,
                     model="new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py"):
    cfg = get_config(model)

    # MODEL WEIGHTS
    cfg.train.init_checkpoint = model_weights

    # DATASETS
    cfg.dataloader.test.dataset.names = ('val',)
    cfg.dataloader.train.dataset.names = ('train',)

    cfg.train.output_dir = output_dir

    # RUN ON CPU
    cfg.train.device = 'cpu'
    # cfg.test.device = 'cpu'
    cfg.model.backbone.bottom_up.norm = "BN"
    cfg.model.backbone.norm = "BN"

    cfg.model.roi_heads.num_classes = num_classes

    cfg.dataloader.train.total_batch_size = 16

    cfg.train.eval_period = 1

    print(LazyConfig.to_py(cfg))

    return cfg


def custom_config_yaml(num_classes, output_dir, model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    # cfg.MODEL.RESNETS.DEPTH = 34
    # cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    # Solver
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = (20, 10000, 20000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 4

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
    return parser


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
     ret = inference_on_dataset(
         model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
     )
     print_csv_format(ret)
     return ret


def main(args, num_classes, output_dir, config_file, model_weights):
    # cfg = LazyConfig.load(config_file)
    cfg = custom_config_py(num_classes, output_dir, model_weights, config_file)
    # default_setup(cfg, args)

    # Eval only
    DatasetCatalog.register('val', lambda d='val': load_data(args.data_dir, d))
    MetadataCatalog.get('val').set(thing_classes=['textblock', 'heading', 'separator'])
    metadata = MetadataCatalog.get('val')
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    cfg.dataloader.evaluator = COCOEvaluator('val', tasks=('bbox', 'segm',), output_dir=output_dir)
    print(do_test(cfg, model))

    # for d in ["train", "val"]:
    #     DatasetCatalog.register(d, lambda d=d: load_data(args.data_dir, d))
    #     MetadataCatalog.get(d).set(thing_classes=["textblock", "heading", "separator"])
    #     # MetadataCatalog.get(d).set(thing_classes=["textblock", "heading"])
    # metadata = MetadataCatalog.get("train")
    #
    # cfg = custom_config_py(num_classes=num_classes,
    #                        output_dir=output_dir,
    #                        model=config_file,
    #                        model_weights=model_weights)
    #
    # # # Predictor
    # # trainer = DefaultTrainer(cfg)
    # # trainer.model = DefaultPredictor(cfg).model
    #
    # cfg.dataloader.evaluator = COCOEvaluator('val', tasks=('bbox',), output_dir=output_dir)
    #
    # # trainer.model.to(cfg.train.device)
    #
    #
    # # default_setup(cfg, args)
    #
    # # model = instantiate(cfg.model)
    # # logger = logging.getLogger("detectron2")
    # # logger.info("Model:\n{}".format(model))
    # # model.to(cfg.train.device)
    #
    # # cfg.optimizer.params.model = model
    # # optim = instantiate(cfg.optimizer)
    #
    # # train_loader = instantiate(cfg.dataloader.train)
    #
    # # model = create_ddp_model(model, **cfg.train.ddp)
    #
    # print("TYPE: ", type(cfg))
    #
    # do_test(cfg, DefaultPredictor(cfg).model)


if __name__ == '__main__':
    args = get_parser().parse_args()

    # for d in ["train", "val"]:
    #     DatasetCatalog.register(d, lambda d=d: load_data(args.data_dir, d))
    #     MetadataCatalog.get(d).set(thing_classes=["textblock", "heading", "separator"])
    # metadata = MetadataCatalog.get("train")

    output_dir = ""
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.data_dir, "output")

    config_file = ""
    if args.config_file is not None:
        config_file = args.config_file
    else:
        config_file = "./new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py"
        # config_file = "/new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py"

    model_weights = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/new_split/par_hd/models/output_x_400ep_lsj/model_0029999.pth"
    num_classes = 3

    num_gpus = 0
    if args.num_gpus is not None:
        num_gpus = args.num_gpus

    launch(main, num_gpus, dist_url=args.dist_url, args=(args, num_classes, output_dir, config_file, model_weights))
