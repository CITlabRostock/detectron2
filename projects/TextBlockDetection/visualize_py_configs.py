from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
import cv2
import os


def visualization(metadata, cfg, image_list_path, model_weights_path, num_classes):
    cfg.train.init_checkpoint = model_weights_path
    cfg.model.roi_heads.box_predictor.test_score_thresh = 0.3
    cfg.model.roi_heads.num_classes = num_classes
    # no equivalent for 'cfg.MODEL.MASK_ON = True', default?

    # necessary to run on CPU
    if "mask_rcnn_R_50_FPN_400ep_LSJ" in model_weights_path:
        cfg.model.backbone.bottom_up.stem.norm = "BN"
        cfg.model.backbone.bottom_up.stages.norm = "BN"
    else:
        cfg.model.backbone.bottom_up.norm = "BN"
    cfg.model.backbone.norm = "BN"

    predictor = DefaultPredictor(cfg)

    with open(image_list_path, 'r') as f:
        image_paths = [l.rstrip() for l in f.readlines()]

    for image_path in image_paths:
        im = cv2.imread(image_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)

        save_folder = "/home/max/tb_det_test/onb173_tb_hd_101_x_lsj_new"
        save_file_name = os.path.basename(image_path)
        # plt.imshow(img)
        # plt.show()
        plt.imsave(os.path.join(save_folder, save_file_name + ".jpg"), img)


if __name__ == "__main__":
    model = "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py"
    model = "new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py"
    # model = "new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.py"
    num_classes = 2

    cfg = get_config(model)
    cfg.train.device = 'cpu'
    # cfg.test.device = 'cpu'x

    model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/old_split/par_hd/models/output_x_400ep_lsj/model_final.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/old_split/par_hd/models/output_x_400ep_lsj/model_0034999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/old_split/par_hd/models/output_mask_rcnn_R_50_FPN_400ep_LSJ/model_final.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/par_hd/models/output_y_400ep_lsj/model_0004999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/par_hd/models/output_y_400ep_lsj/model_0059999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/traindata/par_hd/models/output_x_400ep_lsj/model_0029999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0004999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0014999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0019999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0024999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0029999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0034999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x_lsj/model_0039999.pth"

    image_list_path = "/home/max/data/newseye/gt_data/text_block_detection/" \
                      "GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112/images_val.lst"

    # image_list_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/lists/old_split/images_not_seen.lst"

    # image_list_path = "/home/max/test/onb_detectron_test/images.lst"
    # image_list_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/lists/images_test.lst"
    image_list_path = "/home/max/data/newseye/gt_data/text_block_detection/NewsEye_ONB_173_updated_gt/lists/old_split/images_test.lst"

    CLASSES = ["tb", "hd"]
    # The name of the MetadataCatalog is not important, call it 'train' here
    metadata = MetadataCatalog.get("train").set(thing_classes=CLASSES)

    # get_predictions(cfg, image_list_path, model_weights_path)

    visualization(metadata, cfg, image_list_path, model_weights_path, num_classes)
