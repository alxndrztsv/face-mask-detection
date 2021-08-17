# import libraries;
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# this is the official documentation;
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html?highlight=tfrecord#create-tensorflow-records

# set the paths;
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATIONS_PATH = WORKSPACE_PATH + '/annotations'
IMAGES_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CUSTOM_MODEL_PATH = MODEL_PATH + '/custom_ssd_mobilnet'
CONFIG_FILE = CUSTOM_MODEL_PATH + '/pipeline.config'
CHECKPOINT_PATH = CUSTOM_MODEL_PATH + '/'

# create a list of dictionaries of labels with/without masks;
labels = [{'name': 'mask', 'id': 1}, {'name': 'no mask', 'id': 2}]

# write labels into a txt file:
with open(ANNOTATIONS_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item{\n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# generate tfrecord;
# I used cmd to do that;
# create training data:
# print('python ' + SCRIPTS_PATH + '/generate_tfrecord.py -x ' + IMAGES_PATH + '/train -l ' + ANNOTATIONS_PATH + '/label_map.pbtxt -o ' + ANNOTATIONS_PATH + '/train.record')
# create testing data:
# print('python ' + SCRIPTS_PATH + '/generate_tfrecord.py -x ' + IMAGES_PATH + '/test -l ' + ANNOTATIONS_PATH + '/label_map.pbtxt -o ' + ANNOTATIONS_PATH + '/test.record')

# update pipeline.config for transfer learning;
config = config_util.get_configs_from_pipeline_file(CONFIG_FILE)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_FILE, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# change the number of classes to predict ('Mask', 'NoMask');
pipeline_config.model.ssd.num_classes = 2
pipeline_config.train_config.batch_size = 4
# where to start the training process;
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + \
                                                    '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
# location of label;
pipeline_config.train_input_reader.label_map_path = ANNOTATIONS_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATIONS_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATIONS_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATIONS_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_FILE, 'wb') as f:
    f.write(config_text)

# if happens this error:
# Message type “object_detection.protos.TrainConfig” has no field named “fine_tune_checkpoint_version”
# remove fine_tune_checkpoint_version from pipeline.config

# Used cmd to start training;
# start training model;
# print('python {}/research/object_detection/model_main_tf2.py '
#       '--model_dir={} --pipeline_config_path={}/pipeline.config '
#       '--num_train_steps=5000'.format(APIMODEL_PATH, CUSTOM_MODEL_PATH, CUSTOM_MODEL_PATH))

# start camera.py;
