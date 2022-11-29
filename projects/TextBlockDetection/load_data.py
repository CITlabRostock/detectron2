import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, launch
from detectron2 import model_zoo
from detectron2.config import get_cfg
import argparse


def load_data(data_dir, t="train"):
    if t == "train":
        with open(os.path.join(data_dir, "train.json"), 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open(os.path.join(data_dir, "val.json"), 'r') as file:
            val = json.load(file)
    return val


def custom_config(num_classes, output_dir, model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
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
    cfg.SOLVER.MAX_ITER = 40000
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
                        help="number of GPUs to run training on")
    return parser


def main(num_classes, output_dir, config_file):
    cfg = custom_config(num_classes=num_classes,
                        output_dir=output_dir,
                        model=config_file)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    args = get_parser().parse_args()

    for d in ["train", "val"]:
        DatasetCatalog.register(d, lambda d=d: load_data(args.data_dir, d))
        MetadataCatalog.get(d).set(thing_classes=["textblock", "heading", "separator"])
    metadata = MetadataCatalog.get("train")

    output_dir = os.path.join(args.data_dir, "output")
    if args.output_dir is not None:
        output_dir = args.output_dir

    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    if args.config_file is not None:
        config_file = args.config_file

    num_gpus = 1
    if args.num_gpus is not None:
        num_gpus = args.num_gpus

    launch(main, args.num_gpus)
