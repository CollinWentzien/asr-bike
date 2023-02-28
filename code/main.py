# Main code for ASR Bike
# Created by Collin Wentzien
# Last updated Feb 7 2023
#
# To Run:
# 1. Start stream on Pi with "./start.sh" in home directory
# 2. Start pigpio with "sudo pigpiod"
# 3. Run program "main.py" on Pi
# 4. Wait for calibration, will print "Success" twice
# 5. Change dest_ip in this program to Pi IP (Run "ifconfig" on Pi to find IP)
# 6. Run this program

print("[LOCAL] Loading libraries...")
    
import cv2
import os
import socket
import numpy as np
import tensorflow as tf
from datetime import datetime
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Change to IP of Raspberry Pi
dest_ip = "10.0.0.41"

curr_istr = ""

print("[LOCAL] Libraries loaded successfully")
print("[LOCAL] Loading files...")

# Change CUSTOM_MODEL_NAME to name of tf model
CUSTOM_MODEL_NAME = 'cone_2_7' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# Paths of files that tf uses
paths = {
    'WORKSPACE_PATH': os.path.join('tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('tensorflow','protoc')
 }

# Files that tf accesses on start up
files = {
    'PIPELINE_CONFIG':os.path.join('tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# Change ckpt-x to newest checkpoint per model
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-26')).expect_partial()

# Detect cones
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

print("[LOCAL] Files loaded successfully")
print("[LOCAL] Connecting to bike...")

# Start socket server which will be used to send instructions to the bike
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((dest_ip, 8080))

print("[LOCAL] Connection successful at " + dest_ip)
print("[LOCAL] Starting video capture...")

# Load video stream from bike
# If stuck on "Starting video capture" the stream may have crashed, restart Pi
stream = 'http://' + dest_ip + ':8085/?action=stream'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("[LOCAL] Starting instance")

if not cap.isOpened():
    print('[LOCAL:ERROR] Cannot open RTSP stream (is it running, is there a password?)')
    exit(-1)

while cap.isOpened():
    # Get still frame from video
    ret, frame = cap.read()
    image_np = np.array(frame)

    # Get current time (to see latency between bike and computer)
    now = datetime.now()
    current_time = now.strftime("%M%S")
    
    # Detect cones from frame input
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Duplicate arrays for ease of coding
    boxes = detections['detection_boxes']                        # Location of cones
    max_boxes_to_draw = boxes.shape[0]                           # Max amount of boxes to draw
    scores = detections['detection_scores']                      # Percent probability
    min_score_thresh=.5                                          # Min score / probability for cone to be detected
    count = 0                                                    # Amount of cones detected per frame
    weight = 0                                                   # Found by adding deviation of cones from center

    center = 640 / 2

    dtns = []                                                    # Simplifies detections[] array

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            y, x, y1, x1 = boxes[i]
            score = int((scores[i]) * 100)

            x = (int)(x * 640)
            x1 = (int)(x1 * 640)
            y = (int)(y * 480)
            y1 = (int)(y1 * 480)

            cx = (int)((x + x1) / 2)
            cy = (int)((y + y1) / 2)

            dtns.append({
                "name": "cone_" + str(count),
                "box": (x, y, x1, y1),
                "center": (cx, cy),
                "score": score,
            })

            weight += -(center - cx)

            count += 1

    def get_closest(elem):
        return elem["center"][1]

    dtns.sort(key=get_closest, reverse=True)                     # Closest cone is first in array

    schedule = []                                                # Cones within angle_threst and dist_thresh added here (turn red)
    instruction = []                                             # List of instructions for bike, technically could just be a str or int

    def get_distance(y_height):
        distance = 845805.9*y_height**-1.89103                   # found by taking y points on screen vs actual distance and finding best equation
        return distance

    for detection in dtns:
        angle_thresh = 160                                       # pixels from center that cones will be detected, technically a distance
        dist_thresh = 36                                         # inches from bike when bike will recognize
        critical_dist_thresh = 16                                # will detect within this amount of inches regardless of angle_thresh and avoid
        diff = (detection["center"][0]) - center

        if(get_distance(detection["center"][1]) < critical_dist_thresh):
            # Within critical_dist_thresh

            if(len(schedule) == 0):
                cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (0, 0, 255), 4)
            else:
                cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (20, 20, 255), 2)
            
            schedule.append(detection)
            instruction.append({
                "angle": diff,
                "c": True})
        elif((abs(diff) < angle_thresh and get_distance(detection["center"][1]) < dist_thresh)):
            # Within angle_thresh and within dist_thresh

            if(len(schedule) == 0):
                cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (0, 0, 255), 4)
            else:
                cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (20, 20, 255), 2)
            
            schedule.append(detection)
            instruction.append({
                "angle": diff,
                "c": False})
        else:
            cv2.rectangle(image_np_with_detections, (detection["box"][0], detection["box"][1]), (detection["box"][2], detection["box"][3]), (0, 255, 100), 2)
        
        cv2.putText(image_np_with_detections, detection["name"] + " (" + str(score) + "%)", (detection["box"][0], detection["box"][1] - 20), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(image_np_with_detections, str((int)(get_distance(detection["center"][1]))) + " in.", (detection["box"][0], detection["box"][1] - 6), 0, 0.5, (255, 255, 255), 2)

    # Draw info at top of the screen
    cv2.putText(image_np_with_detections, "count: " + str(count) + ", weight: " + str(weight), (6, 12), 0, 0.5, (255, 255, 255), 2)
    cv2.putText(image_np_with_detections, "order: " + str([i["name"] for i in dtns]), (6, 26), 0, 0.5, (255, 255, 255), 2)
    cv2.putText(image_np_with_detections, "scheduled_order: " + str([i["name"] for i in schedule]), (6, 40), 0, 0.5, (255, 255, 255), 2)

    # Send instructions to bike
    if(len(schedule) > 0):
        if(instruction[0]["c"] == True):
            if(instruction[0]["angle"] > 0):
                msg = current_time + " L"
                cv2.putText(image_np_with_detections, "curr_istr: " + msg + " c", (6, 54), 0, 0.5, (255, 255, 255), 2)
                client.send(msg.encode())
            else:
                msg = current_time + " R"
                cv2.putText(image_np_with_detections, "curr_istr: " + msg + " c", (6, 54), 0, 0.5, (255, 255, 255), 2)
                client.send(msg.encode())
        else:
            if(weight < 0):
                msg = current_time + " L"
                cv2.putText(image_np_with_detections, "curr_istr: " + msg, (6, 54), 0, 0.5, (255, 255, 255), 2)
                client.send(msg.encode())
            else:
                msg = current_time + " R"
                cv2.putText(image_np_with_detections, "curr_istr: " + msg, (6, 54), 0, 0.5, (255, 255, 255), 2)
                client.send(msg.encode())
    else:
        if(len(dtns) > 0):
            msg = current_time + " F"
            cv2.putText(image_np_with_detections, "curr_istr: " + msg, (6, 54), 0, 0.5, (255, 255, 255), 2)
            client.send(msg.encode())
        else:
            msg = current_time + " S"
            cv2.putText(image_np_with_detections, "curr_istr: " + msg, (6, 54), 0, 0.5, (255, 255, 255), 2)
            client.send(msg.encode())

    # Open window on computer
    cv2.imshow('bike (detect:cone)', image_np_with_detections)
    
    # Quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break