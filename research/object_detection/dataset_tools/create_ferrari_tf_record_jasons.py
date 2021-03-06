# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
import sys
sys.path.append('../utils')
import dataset_util
import pickle

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'All_Labels',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
#YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(data,
                       dataset_directory,
                       ignore_difficult_instances=False,
                       image_subdirectory='All_Images'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  #img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, 'SSD_Training_Data','All_Images', data['filename'])
  full_path = full_path.replace('_mp4', '.mp4')
  if '.jpg' not in full_path: full_path = full_path + '.jpg'
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  #poses = []
  difficult_obj = []
  boxes = []

  small_boxes_count = 0

  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      nm = obj['name']
      if nm.lower() == 'Other':
          class_id = 1
      else:
          class_id = 1

      xmin_norm = float(obj['bndbox']['xmin']) / width
      ymin_norm = float(obj['bndbox']['ymin']) / height
      xmax_norm = float(obj['bndbox']['xmax']) / width
      ymax_norm = float(obj['bndbox']['ymax']) / height

      # Skip boxes with size less than: 
      if min(xmax_norm - xmin_norm, ymax_norm - ymin_norm) < 0.008:
        small_boxes_count += 1
        continue

      difficult_obj.append(int(difficult))

      xmin.append(xmin_norm)
      ymin.append(ymin_norm)
      xmax.append(xmax_norm)
      ymax.append(ymax_norm)

      boxes.append([xmin[-1], ymin[-1], xmax[-1], ymax[-1]])

      # classes_text.append(obj['name'].encode('utf8'))
      classes_text.append('ferrari'.encode('utf8'))
      classes.append(class_id)
      truncated.append(int(obj['truncated']))
      #poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      #'image/object/view': dataset_util.bytes_list_feature(poses),
  }))

  return example, boxes, small_boxes_count


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    all_boxes = []
    all_small_boxes_count = 0

    logging.info('Reading from Ferrari dataset.')
    examples_path = os.path.join(data_dir, 'training_split_files', FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, 'SSD_Training_Data', FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example + '.xml')
        try:
            path = path.replace('.mp4','_mp4')
            with tf.gfile.GFile(path, 'rb') as fid:
                xml_str = fid.read()
        except:
            import ipdb
            ipdb.set_trace()
        try:
          xml = etree.fromstring(xml_str)
        except:
          import ipdb
          ipdb.set_trace()
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example, boxes, small_boxes_count = dict_to_tf_example(data, FLAGS.data_dir, FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())
        all_boxes += boxes
        all_small_boxes_count += small_boxes_count

    writer.close()

    import ipdb
    ipdb.set_trace()

    with open('{}_all_boxes.pkl'.format(FLAGS.set), 'wb') as f:
      pickle.dump(all_boxes, f)

if __name__ == '__main__':
  tf.app.run()
