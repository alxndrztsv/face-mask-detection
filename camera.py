# import libraries;
import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# set the paths;
WORKSPACE_PATH = 'Tensorflow/workspace'
ANNOTATIONS_PATH = WORKSPACE_PATH + '/annotations'
MODEL_PATH = WORKSPACE_PATH + '/models'
CUSTOM_MODEL_PATH = MODEL_PATH + '/custom_ssd_mobilnet'
CONFIG_FILE = CUSTOM_MODEL_PATH + '/pipeline.config'
CHECKPOINT_PATH = CUSTOM_MODEL_PATH + '/'

# load train model from checkpoints;
# get a modified pipeline;
configs = config_util.get_configs_from_pipeline_file(CONFIG_FILE)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# restore the latest checkpoint;
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-8')).expect_partial()


# tensorflow function;
@tf.function
def detect(image):
    # resize the image to 320 due to ssd_mobilenet;
    image, shapes = detection_model.preprocess(image)
    # make a prediction;
    prediction = detection_model.predict(image, shapes)
    # postprocess the image;
    detections = detection_model.postprocess(prediction, shapes)
    return detections


# detection in real time;
# create a category index from the label map;
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATIONS_PATH +
                                                                    '/label_map.pbtxt')

# video capture;
cap = cv.VideoCapture(0)
# get width and height;
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:
    # start reading frames;
    ret, frame = cap.read()
    # convert the frame into numpy array;
    image_np = np.array(frame)
    # convert numpy array into tensorflow tensor;
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    # do preprocessing;
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    # start at 1 because the category_index starts from 1;
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    # visualize the detection: draw the box;
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0.3,
        agnostic_mode=False
    )

    # show the real-live detection;
    cv.imshow('mask detection', cv.resize(image_np_with_detections, (400, 300)))

    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
