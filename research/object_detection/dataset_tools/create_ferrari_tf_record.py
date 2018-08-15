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

r"""Convert the Ferrari dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_ferrari_tf_record.py \
        --data_dir=/dataset/CES_2018/SSD_Training_Data/All_Images/ \
        --annotations_dir=/dataset/CES_2018/SSD_Training_Data/All_Labels/ \
        --output_path=/dataset/TF_models/ferrari/
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
import numpy as np
from tqdm import tqdm

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw ferrari dataset.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_float('val_split', 0.2, 'Percent of train to use for validation')

flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       dataset_directory,
                       ignore_difficult_instances=False,
                       image_subdirectory=''):
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
  # img_path = os.path.join(data['folder'], image_subdirectory, data['filename'].split('/')[-1])

  if 'YUN' in data['filename']:
    data['filename'] = data['filename'].replace('_mp4', '.mp4') + '.jpg'

  img_path = data['filename']
  full_path = os.path.join(dataset_directory, img_path)
  
  if not os.path.exists(full_path):
    print("Cant find: {}".format(full_path))
    return None

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
  poses = []
  difficult_obj = []
  names = []

  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      # classes_text.append(obj['name'].encode('utf8'))
      # classes.append(label_map_dict[obj['name']])
      classes_text.append('ferrari'.encode('utf8'))
      classes.append(1)
      truncated.append(int(obj['truncated']))
      # poses.append(obj['pose'].encode('utf8'))
      names.append(obj['name'])

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
      # 'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example, names


def main(_):
  # label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  np.random.seed(0)
  examples_list = [x for x in os.listdir(FLAGS.annotations_dir) if x.endswith('.xml')]

  num_examples = len(examples_list)
  num_train = int(num_examples * (1.0 - FLAGS.val_split))
  num_val = num_examples - num_train

  print("{} Examples total. {} Train -- {} Val".format(num_examples, num_train, num_val))

  split_idxs = {}
  split_idxs['val'] = list(np.random.choice(np.arange(num_examples), num_val))
  split_idxs['train'] = list(set(np.arange(num_examples)).difference(split_idxs['val']))

  all_names = []

  for split in ['train', 'val']:
    output_file = os.path.join(FLAGS.output_path, 'ferrari_{}.record'.format(split))
    
    # writer = tf.python_io.TFRecordWriter(output_file)

    for idx in tqdm(split_idxs[split]):
      example = examples_list[idx]
      path = os.path.join(FLAGS.annotations_dir, example)
      with tf.gfile.GFile(path, 'rb') as fid:
        xml_str = fid.read()

      try:
        xml = etree.fromstring(xml_str)
      except:
        import ipdb
        ipdb.set_trace()

      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      tf_example, names = dict_to_tf_example(data, FLAGS.data_dir,
                                      FLAGS.ignore_difficult_instances)

      all_names.extend(names)

      # if tf_example:
      #   writer.write(tf_example.SerializeToString())

    # writer.close()

    set_names = set(all_names)
    print(set_names)

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
  tf.app.run()
