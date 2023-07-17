
import argparse
import base64
import json
import glob
import os
import os.path as osp
import numpy as np

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )
    """
    oAnnotationDir: 标注信息
    oImageDir: 原始图片
    oVisualDir: 可视化
    jsonFiles: labelme标注的文件夹，格式如下：
    
    dir
    --1.jpg
    --1.json
    --2.jpg
    --2.json
    --...
    --...
    --3.jpg
    --3.json
    """
    oAnnotationDir  = r"G:\segmentation\trainAnno"
    oImageDir       = r"G:\segmentation\trainImage"
    oVisualDir      = r"G:\segmentation\trainVis"
    iAnnotationPath = r"G:\segmentation\OUC\train\*.json"

    jsonFiles = glob.glob(iAnnotationPath)
    jsonFiles = sorted(jsonFiles)

    for path in [oAnnotationDir, oImageDir, oVisualDir]:
        if not osp.exists(path):
            os.makedirs(path)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", default=r'G:\output')
    args = parser.parse_args()

    for idx, json_file in enumerate(jsonFiles):
        if args.out is None:
            out_dir = osp.basename(json_file).replace(".", "_")
            out_dir = osp.join(osp.dirname(json_file), out_dir)
        else:
            out_dir = args.out
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        data = json.load(open(json_file))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        # labelme的标签转成具体的类别数字
        label_name_to_value = {"_background_": 0,               
                            "seaweed": 1,
                            "enteromorpha": 2,
                            "oil": 3,
                            "sea": 4,
                            "river": 5,
                            "land": 6,
                            "spartina": 7, 
                            "suaeda": 8,
                            "tamarix": 9,
                            "reed": 10,
                            "vegetation": 11,
                            "building": 12,
                            "sky": 13,
                            "road": 14,
                            # "person": 15,
                            "boat": 15,
                            # "car": 17,
                            }
        
        """
        if 'person' in cls_name:
            cls_name = '_background_'
        if 'boat' in cls_name:
            cls_name = 'boat'
        if 'car' in cls_name:
            cls_name = '_background_'
        if cls_name == 'anglica':
            cls_name = 'spartina'
        
        utils.shapes_to_label 对于OUC-UAV-SEG数据集来说，做了更改,使用的时候请注意。

        """
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
        )
        print(idx,json_file)
        # 保存图片
        name = "{}/{:05d}.jpg".format(oImageDir, idx)
        PIL.Image.fromarray(img).save( name )

        # 保存标签图片
        name = "{}/{:05d}.png".format(oAnnotationDir, idx)
        PIL.Image.fromarray(lbl.astype(np.uint8)).save( name )

        # 保存可视化图片
        name = "{}/{:05d}.png".format(oVisualDir, idx)
        PIL.Image.fromarray(lbl_viz).save( name )

        
        # 添加语义分割label图片的类别
        path = os.path.join(out_dir,str(idx))
        if not osp.exists(path):
            os.makedirs(path)
        
        PIL.Image.fromarray(img).save(osp.join(path, "img.png"))
        utils.lblsave(osp.join(path, "label.png"), lbl)
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8))            
        lbl_pil.save(osp.join(path, "label1.png"))
        PIL.Image.fromarray(lbl_viz).save(osp.join(path, "label_viz.png"))

        # with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        #     for lbl_name in label_names:
        #         f.write(lbl_name + "\n")

        logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
