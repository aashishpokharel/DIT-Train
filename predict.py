# The predict.py function to be integrated in the AI-Studio
#  python ./dit/object_detection/inference.py --image_path dit/object_detection/publaynet_example.jpeg
# --output_file_name output.jpg --config dit/object_detection/model_final_12k/cascade_dit_base.yaml 
# --opts MODEL.WEIGHTS dit/object_detection/model_final_12k/model_final_12k.pth

import cv2
import json

from ditod import add_vit_config

import torch
from torchvision import transforms
import io


from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
# from PIL import Image # add PIL to requirements if not available
from config import MODEL_PATH, CONFIG_PATH


def predict(X, feature_names = None):
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(CONFIG_PATH)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(MODEL_PATH)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    # img = cv2.imread(args.image_path)
    # if type(X) == 'str':
    #     X = cv2.imread(X)
    X = np.array(Image.open(io.BytesIO(X)).convert("RGB"))
    X = X[:, :, ::-1]
    img = X

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=['Caption', 'Footnote', 'Formula', 'List-item', 'Pagefooter', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text','Title'])

    output = predictor(img)["instances"]
    # print(output)
    
    
    boxes   = [item.tolist() for item in output.get_fields()['pred_boxes']]
    scores  = [item.tolist() for item in output.get_fields()['scores']]
    classes = [item.tolist() for item in output.get_fields()['pred_classes']]
    cropped_images = [img[int(item[1]):int(round(item[3])), int(item[0]):int(round(item[2]))] for item in boxes]

    output  =   {   
                    "scores": list(scores), 
                    "classes":list(classes),
                    "boxes": list(boxes), 
                    # "cropped_images": list(cropped_images)
                }

    # output = json.dumps(output)
    
    return output
if __name__=="__main__":
    from PIL import Image
    import numpy as np
    # img                 = Image.open('test_img.png').convert('RGB')
    with open(filepath, 'rb') as f:
        image = f.read()
    output              = predict(X = image, feature_names = None)
    cv2.imwrite("output.jpg", output['cropped_images'])