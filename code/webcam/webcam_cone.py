print("[LOCAL] Loading libraries")

import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

w = 640
h = 480

print("[LOCAL] Libraries loaded successfully")
print("[LOCAL] Loading files")

CUSTOM_MODEL_NAME = 'cone_2_7' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-26')).expect_partial() # CHANGE CHECKPOINT ****

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

print("[LOCAL] Files loaded successfully")
print("[LOCAL] Starting video capture")

stream = 'http://10.50.72.107:8085/?action=stream'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("[LOCAL] Starting instance")

if not cap.isOpened():
    print('[LOCAL:ERROR] Cannot open RTSP stream (is there a password?)')
    exit(-1)

while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    boxes = detections['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = detections['detection_scores']
    min_score_thresh=.5
    count = 0

    dtns = []

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            y, x, y1, x1 = boxes[i]
            score = int((scores[i]) * 100)

            x = (int)(x * w)
            x1 = (int)(x1 * w)
            y = (int)(y * h)
            y1 = (int)(y1 * h)

            cx = (int)((x + x1) / 2)
            cy = (int)((y + y1) / 2)

            dtns.append({
                "name": "cone_" + str(count),
                "box": (x, y, x1, y1),
                "center": (cx, cy),
                "score": score,
            })

            count += 1

    def get_closest(elem):
        return elem["center"][1]

    dtns.sort(key=get_closest, reverse=True)

    schedule = []
    instruction = []

    def get_distance(y_height):
        distance = 845805.9*y_height**-1.89103
        return distance

    for detection in dtns:
        center = w / 2
        angle_thresh = 160 # technically a distance
        dist_thresh = 36
        critical_dist_thresh = 16
        diff = (detection["center"][0]) - center

        if((abs(diff) < angle_thresh and get_distance(detection["center"][1]) < dist_thresh) or get_distance(detection["center"][1]) < critical_dist_thresh):
            if(len(schedule) == 0):
                cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (0, 0, 255), 4)
            else:
                cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (20, 20, 255), 2)
            schedule.append(detection)
            instruction.append(diff)
        else:
            cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (0, 255, 100), 2)
        
        cv2.putText(image_np_with_detections, detection["name"] + " (" + str(score) + "%)", (detection["box"][0], detection["box"][1] - 20), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(image_np_with_detections, str((int)(get_distance(detection["center"][1]))) + " in.", (detection["box"][0], detection["box"][1] - 6), 0, 0.5, (255, 255, 255), 2)

    cv2.putText(image_np_with_detections, "count: " + str(count), (6, 12), 0, 0.5, (255, 255, 255), 2)
    cv2.putText(image_np_with_detections, "order: " + str([i["name"] for i in dtns]), (6, 26), 0, 0.5, (255, 255, 255), 2)
    cv2.putText(image_np_with_detections, "scheduled_order: " + str([i["name"] for i in schedule]), (6, 40), 0, 0.5, (255, 255, 255), 2)

    cv2.imshow('bike (detect:cone)', image_np_with_detections)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break