import os
import cv2


class Vis2COCO:
    def __init__(self, category_list, is_mode="train"):
        self.category_list = category_list
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode

    def to_coco(self, anno_dir, img_dir):
        self._init_categories()
        img_list = os.listdir(img_dir)
        for img_name in img_list:
            anno_path = os.path.join(anno_dir, img_name.replace(os.path.splitext(img_name)[-1], '.txt'))
            if not os.path.isfile(anno_path):
                print('File is not exist!', anno_path)
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, c = img.shape
            self.images.append(self._image(img_path, h, w))
            if self.img_id % 500 == 0:
                print("处理到第{}张图片".format(self.img_id))

            with open(anno_path, 'r') as f:
                for lineStr in f.readlines():
                    try:
                        if ',' in lineStr:
                            xmin, ymin, w, h, score, category, trunc, occlusion = lineStr.split(',')
                        else:
                            xmin, ymin, w, h, score, category, trunc, occlusion = lineStr.split()
                    except:
                        # print('error: ', anno_path, 'line: ', lineStr)
                        continue
                    if int(category) in [0, 11] or int(w) < 4 or int(h) < 4:
                        continue
                    label, bbox = int(category) - 1, [int(xmin), int(ymin), int(w), int(h)]
                    annotation = self._annotation(label, bbox)
                    self.annotations.append(annotation)
                    self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['categories'] = self.categories
        instance['annotations'] = self.annotations
        instance['images'] = self.images

        return instance

    def _init_categories(self):
        cls_num = len(self.category_list)
        for v in range(0, cls_num):
            # print(v)
            category = {}
            category['id'] = v
            category['name'] = self.category_list[v]
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['file_name'] = os.path.basename(path)
        image['id'] = self.img_id
        image['width'] = w
        image['height'] = h
        return image

    def _annotation(self, label, bbox):
        area = bbox[2] * bbox[3]
        annotation = {}
        annotation['area'] = area
        annotation['bbox'] = bbox
        annotation['category_id'] = label
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['iscrowd'] = 0

        return annotation

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=4, separators=(',', ': '))


def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cvt_vis2coco(img_path, anno_path, save_path, train_ratio=0.9, category_list=[], mode='train'):  # mode: train or val
    vis2coco = Vis2COCO(category_list, is_mode=mode)
    instance = vis2coco.to_coco(anno_path, img_path)
    if not os.path.exists(os.path.join(save_path, "Anno")):
        os.makedirs(os.path.join(save_path, "Anno"))
    vis2coco.save_coco_json(instance,
                            os.path.join(save_path, 'Anno', 'instances_{}2017.json'.format(mode)))
    print('Process {} Done'.format(mode))


if __name__ == "__main__":
    """
    路径如下：
    ├─VisDrone2019-DET-test-challenge
    │  ├─images
    ├─VisDrone2019-DET-test-dev
    │  ├─annotations
    │  ├─images
    ├─VisDrone2019-DET-train
    │  ├─annotations
    │  ├─images
    └─VisDrone2019-DET-val
        ├─annotations
        ├─images
    """
    root_path = r'D:\data\detection\VisDrone'
    category_list = ['pedestrain', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
                     'motor']
    for mode in ['val']:
        cvt_vis2coco(os.path.join(root_path, 'VisDrone2019-DET-{}/images'.format(mode)),
                     os.path.join(root_path, 'VisDrone2019-DET-{}/annotations'.format(mode)),
                     root_path, category_list=category_list, mode=mode)  # mode: train or val
