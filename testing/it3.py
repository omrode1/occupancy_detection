from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import torch
import time
from datetime import datetime
import csv
import os
import sys

#define ips
ip = "rtsp://admin:admin123@10.0.22.205:554/cam/realmonitor?channel=1&subtype=0"

#define rois for A1 and A2 cubicle areas
# roi=[66, 96, 1224, 688] #A1
# roi1=[70, 98, 598,685] #A2

roi =[1, 49, 475, 665] #a5
roi1  = [597, 38, 1240, 671] #a6

#class names for the model
names = {
    0: 'worker'
}

#defineing shifts for the day
shifts = {
	'A': ["06:00", "14:30"],
	'B': ["14:30", "23:00"],
	'C': ["23:00", "06:00"]
}


#detecting the shift
def get_shift():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    for shift, time in shifts.items():
        start = time[0]
        end = time[1]
        if current_time >= start and current_time < end:
            return shift
    return None

#detecting the area from matching camera ips with the area
camera_ips = { 'A1/A2' : ["rtsp://admin:admin123@10.0.22.205:554/cam/realmonitor?channel=1&subtype=0"],
               'A5/A6' : ["rtsp://admin:admin123@10.0.22.190:554/cam/realmonitor?channel=1&subtype=0"]
}

def get_area(ip):
    for area, ips in camera_ips.items():
        if ip in ips:
            return area
    return None

#detecting the camera ip from the camera name
def get_camera_ip(camera_name):
    for area, ips in camera_ips.items():
        if camera_name in area:
            return ips[0]
    return None

#detecting the camera name from the camera ip
def get_camera_name(ip):
    for area, ips in camera_ips.items():
        if ip in ips:
            return area
    return None




#ininializing the csv file for the data
def init_csv():
    with open('data.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['camera_name', 'area', 'shift', 'time', 'count', 'In/Out'])


#check vacancy of roi
def check_vacancy(area):
    camera_name = get_camera_name(ip)
    with open('data.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == camera_name and row[1] == area and row[5] == "IN":
                return False
    return True


#update the entry time of the worker
def update_entry_time(area, shift):
    camera_name = get_camera_name(ip)
    with open('data.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([camera_name, area, shift, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, "IN"])


#update the exit time of the worker
def update_exit_time(area, shift):
    camera_name = get_camera_name(ip)
    with open('data.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([camera_name, area, shift, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, "OUT"])
        

class detect_tray:
    """
    Class to detect workers in a video stream and track their presence in specific regions of interest (ROIs).
    """

    def __init__(self):
        self.model = YOLO("worker4.pt")
        self.worker_in_roi = {'A1': False, 'A2': False}  # Flags to track worker presence in ROIs

    def draw_boxes(self, image, bbox, class_label):
        
        
        """
        Draw bounding boxes around detected objects on the given image.

        Args:
            image (numpy.ndarray): The input image.
            bbox (list): The bounding box coordinates [x, y, width, height].
            class_label (str): The class label of the detected object.

        Returns:
            numpy.ndarray: The image with bounding boxes drawn.
        """
        x, y, width, height = bbox
        color = (0, 255, 0)
        cv2.rectangle(image, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), color, 2)
        cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def process_camera(self):
        """
        Process the video stream from the camera, perform object detection, and track worker presence in ROIs.
        """
        video_stream = cv2.VideoCapture("rtsp://admin:admin123@10.0.22.190:554/cam/realmonitor?channel=1&subtype=0")
        frame_number = 0
        last_detection_time = time.time()

        
        if not video_stream.isOpened():
            print("Error: Video stream not opened.")
            return

        while True:
            try:
                ret, frame = video_stream.read()
                if not ret:
                    print("Error reading frame: ret =", ret)
                    break

                # Resize frame to 1280x720
                frame = cv2.resize(frame, (1280, 720))
                
                #Perform object detection
                predictions = self.model.predict(frame, conf=0.50)

                
                # Process detections
                person_detected = False
                for result in predictions:
                    boxes = result.boxes
                    xywh = boxes.xywh
                    classes = boxes.cls

                    for b, c in zip(xywh, classes):
                        b = b.tolist()
                        b = [round(num) for num in b]
                        x, y, width, height = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        class_label = names[int(c)]

                        if class_label == 'worker':
                            cubicle = 'A1' if (roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]) else 'A2'
                            if not self.worker_in_roi[cubicle]:
                                self.worker_in_roi[cubicle] = True
                                shift = get_shift()  # Replace with your function to determine the shift
                                update_entry_time(cubicle, shift)
                        else:
                            cubicle = 'A1' if (roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]) else 'A2'
                            if self.worker_in_roi[cubicle]:
                                self.worker_in_roi[cubicle] = False
                                shift = get_shift()  # Replace with your function to determine the shift
                                update_exit_time(cubicle, shift)

                        
                        # Draw bounding box
                        frame = self.draw_boxes(frame, b, class_label)
                
                    #draw ROI on frame
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.rectangle(frame, (roi1[0], roi1[1]), (roi1[2], roi1[3]), (0, 255, 0), 2)

                # Display frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print("Exception:", e)
                import traceback
                traceback.print_exc()
                pass

                print("Exception:", e)
                import traceback
                traceback.print_exc()

        video_stream.release()
        cv2.destroyAllWindows()

# Initialize object
detector = detect_tray()

# Start processing camera
detector.process_camera()
