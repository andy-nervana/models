import os
import sys
from lxml import etree
sys.path.append('../utils')
import dataset_util
import tensorflow as tf
from dicttoxml import dicttoxml

label_dir = '/dataset/CES_2018/SSD_Training_Data/All_SSD/All_Labels/'
img_dir = '/dataset/CES_2018/SSD_Training_Data/All_SSD/All_Images/'

label_files = os.listdir(label_dir)

for lf in label_files:
    if 'TURN' in lf:
        # image_f = os.path.join(img_dir, lf.split('.xml')[0] + '.jpg')
        label_f = os.path.join(label_dir, lf)
        
        with tf.gfile.GFile(label_f, 'rb') as fid:
            xml_str = fid.read()

        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        # im = imread(image_f)

        width = int(data['size']['width'])
        height = int(data['size']['height'])

        if width == 3800 and height == 2100:
            with open(label_f, 'r') as f:
                xml = f.readlines()

            xml = xml[0].replace('<width>3800</width>', '<width>3840</width>')
            xml = xml.replace('<height>2100</height>', '<height>2160</height>')

            with open(label_f, 'w') as f:
                f.write(xml)