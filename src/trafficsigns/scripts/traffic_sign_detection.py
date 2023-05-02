import cv2 as cv
import numpy as np
from skimage import exposure
from keras.models import load_model
from trafficsign import TrafficSign

GLOBAL_LABELS = ['prohibitory',
                'danger',
                'mandatory',
                'other']

SIGN_TYPES_IDS = [
    'speed-20',
    'speed-30', 
    'speed-50', 
    'speed-60', 
    'speed-70', 
    'speed-80', 
    'end-speed-80', 
    'speed-100', 
    'speed-120', 
    'no-passing', 
    'no-passing-3ton5', 
    'punctual-priority', 
    'road-priority', 
    'yield', 
    'stop', 
    'no-vehicles', 
    'no-vehicles-3ton5', 
    'no-entry', 
    'caution', 
    'dangerous-curve-left', 
    'dangerous-curve-right', 
    'double-curve', 
    'bumpy-road', 
    'slippery-road', 
    'road-narrows-right', 
    'road-work', 
    'traffic-signals', 
    'pedestrians', 
    'children-crossing', 
    'bicycles-crossing', 
    'danger-ice-snow',
    'wild-animals-crossing', 
    'end-speed-no-passing', 
    'right-only', 
    'left-only', 
    'ahead-only', 
    'straight-right-only', 
    'straight-left-only', 
    'keep-right', 
    'keep-left', 
    'roundabout', 
    'end-no-passing', 
    'end-no-passing-3ton5' 
]

SPECIFIC_LABELS = [
    'Speed limit (20km/h)',
    'Speed limit (30km/h)', 
    'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 
    'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 
    'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 
    'No passing', 
    'No passing veh over 3.5 tons', 
    'Right-of-way at intersection', 
    'Priority road', 
    'Yield', 
    'Stop', 
    'No vehicles', 
    'Veh > 3.5 tons prohibited', 
    'No entry', 
    'General caution', 
    'Dangerous curve left', 
    'Dangerous curve right', 
    'Double curve', 
    'Bumpy road', 
    'Slippery road', 
    'Road narrows on the right', 
    'Road work', 
    'Traffic signals', 
    'Pedestrians', 
    'Children crossing', 
    'Bicycles crossing', 
    'Beware of ice/snow',
    'Wild animals crossing', 
    'End speed + passing limits', 
    'Turn right ahead', 
    'Turn left ahead', 
    'Ahead only', 
    'Go straight or right', 
    'Go straight or left', 
    'Keep right', 
    'Keep left', 
    'Roundabout mandatory', 
    'End of no passing', 
    'End no passing veh > 3.5 tons' 
]


class TrafficSignDetector(object):
    def __init__(self):
        self.yolov4_model = cv.dnn.readNet("yolov4-tiny_training_last.weights", "yolov4-tiny_training.cfg") # DETECTION MODEL
        self.recognition_model = load_model('DeepLeNet-5_CLAHE_AUG(v2).h5') # RECOGNITION MODEL

        #get last layers names
        self.layer_names = self.yolov4_model.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolov4_model.getUnconnectedOutLayers()]
        self.confidence_threshold = 0.3
        self.font = cv.FONT_HERSHEY_SIMPLEX


    def get_traffic_signs(self, img):
        # Detecting objects (YOLO)
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolov4_model.setInput(blob)
        outs = self.yolov4_model.forward(self.output_layers)

        height, width, _ = img.shape

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        traffic_signs = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                global_label = str(GLOBAL_LABELS[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                
                crop_img = img[y:y+h, x:x+w]
                
                if len(crop_img) >0:
                    crop_img = cv.resize(crop_img, (32, 32))
                    
                    img_norm = exposure.equalize_adapthist(crop_img, clip_limit=0.1).astype('float32')
                    
                    prediction = self.recognition_model.predict(np.array([img_norm]))
                    
                    selected_type = np.argmax(prediction[0])
                    label_specific = SPECIFIC_LABELS[selected_type]

                    traffic_signs.append(TrafficSign(category=GLOBAL_LABELS[class_ids[i]], type=SIGN_TYPES_IDS[selected_type], label=label_specific,
                                                     x=x, y=y, width=w, height=h, confidence=confidences[i]))
                    
                    img = cv.putText(img, label_specific, (x, y-10), self.font, 0.5, (0,0,255), 2)
                    
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                img = cv.putText(img, global_label, (x, y+h+15), self.font, 0.5, (0,0,255), 2)
                
        return img, traffic_signs