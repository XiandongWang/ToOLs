import os
import xml.etree.ElementTree as ET
import cv2



def convert_labels(label_dir, image_dir, out_dir):
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        image_file = os.path.join(image_dir, label_file.replace('.txt', '.jpg'))
        print(image_file)
        if not os.path.isfile(image_file):
            continue
        
        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = os.path.basename(os.path.normpath(image_dir))
        filename = ET.SubElement(root, "filename")
        filename.text = os.path.basename(image_file)
        path = ET.SubElement(root, "path")
        path.text = image_file

        size = ET.SubElement(root, "size")
        img = cv2.imread(image_file)
        height, width, channels = img.shape
        w = ET.SubElement(size, "width")
        w.text = str(width)
        h = ET.SubElement(size, "height")
        h.text = str(height)
        d = ET.SubElement(size, "depth")
        d.text = str(channels)

        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f.readlines():
                label = line.strip().split()
                label_name = label[0]
                x_center, y_center, w, h = float(label[1])*width, float(label[2])*height, float(label[3])*width, float(label[4])*height
                object_xml = ET.SubElement(root, "object")
                name = ET.SubElement(object_xml, "name")
                name.text = label_name
                pose = ET.SubElement(object_xml, "pose")
                pose.text = "Unspecified"
                truncated = ET.SubElement(object_xml, "truncated")
                truncated.text = "0"
                difficult = ET.SubElement(object_xml, "difficult")
                difficult.text = "0"

                bndbox = ET.SubElement(object_xml, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(int((x_center - w / 2)))
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(int((y_center - h / 2)))
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(int((x_center + w / 2)))
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(int((y_center + h / 2)))

        xml_file = os.path.join(out_dir, label_file.replace('.txt', '.xml'))
        tree = ET.ElementTree(root)
        tree.write(xml_file)
        
if __name__=='__main__':

    label_dir = r"G:\detection\VisDrone\VisDrone2019-DET-val\labels"
    image_dir = r"G:\detection\VisDrone\VisDrone2019-DET-val\images"
    out_dir = r"G:\detection\VisDrone\xml"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    convert_labels(label_dir,image_dir,out_dir)