import time 
from datetime import datetime
import os
import sys
import cv2
import numpy as np
import boto3
from ultralytics import YOLO


#connect to s3 bucket
client = boto3.client('s3',
                      aws_access_key_id ="",
                      aws_secret_access_key ="",
                      region_name = "us-east-1")

def upload_file_to_s3(file, bucket_name, st, acl="public-read"):
    try:
        Path_of_upload = "PPE_IMAGES/" + st
        print(Path_of_upload)

        client.put_object(Bucket=bucket_name, Key = Path_of_upload, Body=file,ACL =acl)
        return "Successful"
    except:
        return "Failed"

#define shifts of the day
shifts = {
	'A': ["06:00", "14:30"],
	'B': ["14:30", "23:00"],
	'C': ["23:00", "06:00"]
}


#function to check the current shift
def check_shift():
    """
    Check the current time and determine the corresponding shift.

    Returns:
        str: The name of the shift.

    """
    current_time = datetime.now().strftime("%H:%M")
    for shift, time in shifts.items():
        if time[0] <= current_time <= time[1]:
            return shift
        
#init csv file (only for testing)
def init_csv():
    with open('PPE.csv', 'w') as f:
        f.write('Date,Time,Shift,Image\n')

#csv file will have following columns 
#date, cubicle number, shift, intime , outtime, image
def write_csv(date, time, shift, img):
    with open('PPE.csv', 'a') as f:
        f.write(f'{date},{time},{shift},{img}\n') 

#roi values for the cubicle
roi=[66, 96, 1224, 688]
roi1=[70, 98, 598,685]   #only for testing purpose

#master roi 
roi_master = {
    'A1' : [66, 96, 1224, 688],
    'A2' : [66, 96, 1224, 688],
    'A3' : [66, 96, 1224, 688],
    'A4' : [66, 96, 1224, 688],
    'A5' : [66, 96, 1224, 688],
    'A6' : [66, 96, 1224, 688],
    'A7' : [66, 96, 1224, 688],
    'A8' : [66, 96, 1224, 688],
    'A9' : [66, 96, 1224, 688],
    'A10' : [66, 96, 1224, 688],
    'A11' : [66, 96, 1224, 688],
    'A12' : [66, 96, 1224, 688],
    'A13' : [66, 96, 1224, 688],
    'A14' : [66, 96, 1224, 688],
    
}


#camera ip stremas for the cubicles
stream = {"rtsp://admin:admin123@10.0.22.205:554/cam/realmonitor?channel=1&subtype=0"} #for either 1 or 2 cubicles


#there are two cubicles within a single stream
master_stream = {
    'A1', 'A2' : "rtsp://admin:admin123@10.0.22.205:554/cam/realmonitor?channel=1&subtype=0",
}


#class to detect in camera feed 
names = {
    0: 'worker'
}


#function to update intime, outtime , shift and cubicle 
def update_intime_outtime(cubicle, shift, intime, outtime):
    with open('PPE.csv', 'r') as f:
        data = f.readlines()
    with open('PPE.csv', 'w') as f:
        f.write('Date,Time,Shift,Intime,Outtime,Image\n')
        for line in data:
            if cubicle in line:
                line = line.split(',')
                line[2] = shift
                line[3] = intime
                line[4] = outtime
                line = ','.join(line)
            f.write(line)

class detect_tray:
    def __init__(self):
        self.model = YOLO("worker3.pt")
        self.worker_in_cubicle = {'A1': False, 'A2': False, 
                                  'A3': False, 'A4': False, 
                                  'A5': False, 'A6': False, 
                                  'A7': False, 'A8': False, 
                                  'A9': False, 'A10': False,
                                  'A11': False, 'A12': False, 
                                  'A13': False, 'A14': False }

    def process_camera(self):
        """
        Process the camera stream to detect workers in each cubicle.

        This method reads frames from the camera stream, resizes them, and applies object detection
        to detect workers in each cubicle. It updates the `worker_in_cubicle` dictionary based on
        the detection results.

        Returns:
            None
        """
        for cubicle, stream in master_stream.items():
            cap = cv2.VideoCapture(stream)
            if not cap.isOpened():
                print("Error: Could not open video.")
                return
            
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame.")
                        break

                    frame = cv2.resize(frame, (1280, 720))
                    predictions = self.model.predict(frame, conf=0.5)
                    frame = frame[roi_master[cubicle][1]:roi_master[cubicle][3], roi_master[cubicle][0]:roi_master[cubicle][2]]
                    results = self.model(frame)

                    if len(results.xyxy[0]) > 0:
                        self.worker_in_cubicle[cubicle] = True
                    else:
                        self.worker_in_cubicle[cubicle] = False
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Error: {str(e)}") 

            #draw rectangle around the detected object and demarcate the roi boxes on the frame

            cv2.rectangle(frame, (roi_master[cubicle][0], roi_master[cubicle][1]), (roi_master[cubicle][2], roi_master[cubicle][3]), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cap.release()

            #bounding box around the detected worker 
            for i in range(len(results.xyxy[0])):
                x1, y1, x2, y2 = results.xyxy[0][i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()


                




