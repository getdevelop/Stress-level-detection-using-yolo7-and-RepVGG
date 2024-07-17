
## Setup
### pip
`pip install -r requirements.txt`
### conda
`conda env create -f env.yaml`
## Emotions
This model detects 8 basic facial expressions:
- anger
- contempt
- disgust
- fear
- happy
- neutral
- sad
- surprise<br>
and then attempts to assign them appropriate colours. It classifies every face, even if it is not that confident about the result!
## Usage
```
usage: main.py [-h] [--source SOURCE] [--img-size IMG_SIZE] [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]
               [--device DEVICE] [--hide-img] [--output-path OUTPUT_PATH | --no-save] [--agnostic-nms] [--augment]
               [--line-thickness LINE_THICKNESS] [--hide-conf] [--show-fps]

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       source
  --img-size IMG_SIZE   inference size (pixels)
  --conf-thres CONF_THRES
                        face confidence threshold
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --hide-img            hide results
  --output-path OUTPUT_PATH
                        save location
  --no-save             do not save images/videos
  --agnostic-nms        class-agnostic NMS
  --augment             augmented inference
  --line-thickness LINE_THICKNESS
                        bounding box thickness (pixels)
  --hide-conf           hide confidences
  --show-fps            print fps to console
```
## Architecture
There are two parts to this code: facial detection and emotion classification