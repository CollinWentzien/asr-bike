# (Sorta) Self-Driving Bicycle
## ASR-1 Project by Collin Wentzien

> Created      30 November 2022<br>
> Last Updated 28 February 2023

### Stream
- /usr/local/bin/mjpg_streamer -i '/usr/local/lib/mjpg-streamer/input_uvc.so -n -f 20 -r 640x480' -o '/usr/local/lib/mjpg-streamer/output_http.so -p 8085 -w /urs/local/share/mjpg-streamer/www/'
- http://10.50.72.107:8085/?action=stream


### Resources
- ssh bike@bike:1725
- mjpg-streamer GitHub: https://github.com/jacksonliam/mjpg-streamer
- imagezmq: https://pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/ (ha not using this)
- Using mjpg-streamer: https://www.sigmdel.ca/michel/ha/rpi/streaming_en.html
- LabelStudio: cmd > label-studio ; localhost:8080/
- Activate venv: source bike/bin/activate **OR** .\bike\Scripts\activate
- Run training (old): python tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=2000
- Tensorboard: \models\my_ssd_mobnet_2\train (loss metrics) **OR** \models\my_ssd_mobnet_2\eval (evaluation metrics): tensorboard --logdir=.
- NOTE: if the training isnt running then make sure the record file size != 0 (did not take *hours* of troubleshooting)
- sudo pigpiod
- https://github.com/nicknochnack/TFODCourse

### Other
- distance_points: (12,360),(24,264),(36,205),(48,174),(60,157),(72,141),(84,130),(96,123)
    - this is assuming that the camera angle doesn't change...