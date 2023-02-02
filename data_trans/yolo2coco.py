import os
import json
from PIL import Image
from tqdm import tqdm


def main():
    """
    1. 需要更改的四处地方coco_format_save_path、yolo_format_annotation_path、img_pathDir和class_names
    2. image id 和 annotation id 以及类别id 均从0开始
    3. 当前image、categories、annotations三个key是必须要有的，其他的key值可以不存在，不影响实验结果。
    :return: json
    """
    coco_format_save_path = r'D:\data\detection\VisDrone\VisDrone2019-DET-val'  # 要生成的标准coco格式标签所在文件夹
    yolo_format_annotation_path = r'D:\data\detection\VisDrone\VisDrone2019-DET-val\labels'  # yolo格式标签所在文件夹
    img_pathDir = r'D:\data\detection\VisDrone\VisDrone2019-DET-val\images'  # 图片所在文件夹

    categories = []
    class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                   'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    for label in class_names:
        categories.append({'id': class_names.index(label), 'name': label})

    write_json_context = dict()
    write_json_context['categories'] = categories
    write_json_context['annotations'] = []
    write_json_context['images'] = []

    imageFileList = os.listdir(img_pathDir)
    img_id = 0
    anno_id = 0

    for i, imageFile in tqdm(enumerate(imageFileList)):
        imagePath = os.path.join(img_pathDir, imageFile)
        image = Image.open(imagePath)
        W, H = image.size
        img_context = {}
        img_context['file_name'] = imageFile
        img_context['id'] = img_id
        img_context['width'] = W
        img_context['height'] = H
        write_json_context['images'].append(img_context)
        txtFile = imageFile.split('.')[0] + '.txt'
        with open(os.path.join(yolo_format_annotation_path, txtFile), 'r') as fr:
            lines = fr.readlines()
        for j, line in enumerate(lines):
            bbox_dict = {}

            class_id, x, y, w, h = line.strip().split(' ')
            class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)

            xmin = (x - w / 2) * W
            ymin = (y - h / 2) * H
            xmax = (x + w / 2) * W
            ymax = (y + h / 2) * H
            width = max(0, xmax - xmin)
            height = max(0, ymax - ymin)

            bbox_dict['area'] = height * width
            bbox_dict['bbox'] = [xmin, ymin, width, height]
            bbox_dict['category_id'] = class_id
            bbox_dict['id'] = anno_id
            bbox_dict['image_id'] = img_id
            bbox_dict['iscrowd'] = 0

            write_json_context['annotations'].append(bbox_dict)
            anno_id += 1
        img_id += 1
    name = os.path.join(coco_format_save_path, "annotations" + '.json')
    with open(name, 'w') as fw:
        json.dump(write_json_context, fw, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
