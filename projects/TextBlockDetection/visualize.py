from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
import cv2
import os


def get_predictions(cfg, image_list_path, model_weights_path):
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    predictor = DefaultPredictor(cfg)

    with open(image_list_path, 'r') as f:
        image_paths = [l.rstrip() for l in f.readlines()]

    prediction_dict = {}
    for image_path in image_paths:
        im = cv2.imread(image_path)
        outputs = predictor(im)
        print(type(outputs["instances"]))
        instances = outputs["instances"]
        fields = instances.get_fields()

        prediction_dict[image_path] = {}

        prediction_dict['pred_boxes'] = fields['pred_boxes'].tensor.detach().cpu().numpy()  # Nx4, each row (x1,y1,x2,y2)
        for k in ['scores', 'pred_classes', 'pred_masks']:  # Nx1, Nx1, NxHxW resp.
            prediction_dict[k] = fields[k].detach().cpu().numpy()

        print(prediction_dict['pred_boxes'].shape)
        print(prediction_dict['pred_boxes'])
        print(prediction_dict['scores'].shape)
        print(prediction_dict['scores'])
        print(prediction_dict['pred_classes'].shape)
        print(prediction_dict['pred_classes'])
        print(prediction_dict['pred_masks'].shape)

        exit(0)


def visualization(metadata, cfg, image_list_path, model_weights_path):
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    predictor = DefaultPredictor(cfg)

    with open(image_list_path, 'r') as f:
        image_paths = [l.rstrip() for l in f.readlines()]

    for image_path in image_paths:
        im = cv2.imread(image_path)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)

        save_folder = "/home/max/tb_det_test/tb_hd_sep_101"
        save_file_name = os.path.basename(image_path)
        plt.imsave(os.path.join(save_folder, save_file_name), img)


if __name__ == "__main__":
    cfg = get_cfg()
    # cfg.merge_from_file(
    #     "/home/max/devel/src/git/python/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.merge_from_file(
    #     "/home/max/devel/src/git/python/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file(
        "/home/max/devel/src/git/python/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112/models/model_final_101.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/model_0044999.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/50/model_final_50.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101/model_final_101.pth"
    # model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101/model_0014999.pth"
    model_weights_path = "/home/max/data/newseye/gt_data/text_block_detection/GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112_hd_sep/models/101_x/model_final_x.pth"
    image_list_path = "/home/max/data/newseye/gt_data/text_block_detection/" \
                      "GT_article_koelnische_zeitung_1936_-_reduziert/traindata/koeln112/images_val.lst"

    image_list_path = "/home/max/test/onb_detectron_test/images.lst"

    metadata = None

    # get_predictions(cfg, image_list_path, model_weights_path)

    visualization(metadata, cfg, image_list_path, model_weights_path)
