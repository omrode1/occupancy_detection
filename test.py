import psycopg2 as spg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time
from datetime import datetime
import cv2
import torch
import numpy as np
from threading import Thread
import sys
import urllib
import json
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
import boto3
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from module_name import connect
from utils.general import plot_one_box





#define shifts for the day
shifts = {
    'A': ["06:00", "14:30"],
    'B': ["14:30", "23:00"],
    'C': ["23:00", "06:00"] 
}

#detecting the shift
def inShift(currentTime):
	
	if shifts['A'][0] <= currentTime < shifts['A'][1]:
		return 'A'
	elif shifts['B'][0] <= currentTime < shifts['B'][1]:
		return 'B'
	return 'C'

#detecting the area from matching camera ips with the area
def inRegion(c1, c2, rois):
    cx = (c1[0] + c2[0]) // 2
    cy = (c1[1] + c2[1]) // 2
    for i in range(len(rois)):
        roi = rois[i]
        if roi[0] < cx < roi[2] and roi[1] < cy < roi[3]:
            return i
    return None

#close the entries in the database
def closeEntries():
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("SELECT * FROM entries WHERE exit_time IS NULL")
    rows = cur.fetchall()
    for row in rows:
        cur.execute("UPDATE entries SET exit_time = %s WHERE id = %s", (datetime.now(), row[0]))
        conn.commit()
    cur.close()
    conn.close()

#detecting the camera ip from the camera name
def getCameraIp(cameraName):
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("SELECT * FROM cameras WHERE name = %s", (cameraName,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    return row[2]

#detecting the camera name from the camera ip
def getCameraName(cameraIp):
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("SELECT * FROM cameras WHERE ip = %s", (cameraIp,))
    row = cur.fetchone()

    cur.close()
    conn.close()

    return row[1]

#check vacancy of roi
def checkVacancy(area, cameraIp):
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("SELECT * FROM entries WHERE camera_ip = %s AND area = %s AND exit_time IS NULL", (cameraIp, area))
    row = cur.fetchone()
    cur.close()
    conn.close()

    return row is None

#ininializing the csv file for the data
def initCsv():
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )


    cur = conn.cursor()
    
    cur.execute("CREATE TABLE IF NOT EXISTS entries (id SERIAL PRIMARY KEY, camera_ip VARCHAR(255), area VARCHAR(255), shift VARCHAR(255), entry_time TIMESTAMP, exit_time TIMESTAMP)")
    conn.commit()
    cur.close()
    conn.close()

#update the entry time of the worker
def updateEntryTime(area, shift, cameraIp):
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("INSERT INTO entries (camera_ip, area, shift, entry_time) VALUES (%s, %s, %s, %s)", (cameraIp, area, shift, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()

#update the exit time of the worker
def updateExitTime(area, shift, cameraIp):
    conn = spg.connect(
        database="postgres",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute("SELECT * FROM entries WHERE camera_ip = %s AND area = %s AND exit_time IS NULL", (cameraIp, area))
    row = cur.fetchone()
    cur.execute("UPDATE entries SET exit_time = %s WHERE id = %s", (datetime.now(), row[0]))
    conn.commit()
    cur.close()
    conn.close()

class detect_tray:
    """
    Class for detecting trays in a video stream using YOLOv5 object detection model.

    Attributes:
        productionHouse (str): The production house name.
        streamLink (str): The link to the video stream.
        areaName (str): The name of the area where the detection is performed.
        rois (list): List of regions of interest (ROIs) for detection.
        con (connection): Connection object for database connection.
        cur (cursor): Cursor object for executing database queries.
        device (str): Device used for model inference.
        model (YOLOv5 model): YOLOv5 object detection model.
        names (list): List of class names for the model.
        stride (int): Stride value for the model.
        cap (cv2.VideoCapture): Video capture object for reading frames from the stream.
        thread_cap (Thread): Thread object for starting the input stream.
        prev_frame_time (float): Previous frame time for calculating FPS.
        areaType (str): Type of the area (e.g., explosive, non-explosive).
        areaData (dict): Dictionary containing area-specific data.
        plant (str): The name of the plant.
        model_exp (YOLOv5 model): YOLOv5 object detection model for explosive detection.
        names_exp (list): List of class names for the explosive detection model.
        stride_exp (int): Stride value for the explosive detection model.
        processing_device (str): Device used for model inference in the explosive detection model.
        capOpened (bool): Flag indicating if the video capture is opened.
        frame (numpy.ndarray): Current frame from the video stream.

    Methods:
        init_explosive_parameters: Initialize explosive detection parameters.
        start_input: Start the input stream.
        init_capture: Initialize the video capture.
        update_frame: Update the current frame.
        read: Read the current frame.
        run_yolov5s_exp: Run the explosive detection model on the frame.
        run_yolov5s: Run the object detection model on the frame.
        process: Process the frame using the object detection model.
        process_exp: Process the frame using the explosive detection model.
        process_camera: Process the frame based on the area type.
    """
    def __init__(self):
        self.productionHouse = 'EEL'
        self.streamLink = 'rtsp://admin:admin123@'
        self.areaName = 'A1/A2'
        self.rois = [[70, 98, 598, 685]]
        self.con, self.cur = connect()
        self.init_csv()
        self.device = select_device()
        self.model = attempt_load('weights/happy_27_05_22.pt', map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.stride = int(self.model.stride.max())
        self.cap = cv2.VideoCapture(self.streamLink)
        self.thread_cap = Thread(target=self.start_input, daemon=True)
        self.thread_cap.start()
        self.prev_frame_time = 0
        self.areaType = 'explosive'
        self.init_explosive_parameters()
        self.areaData = {
            "streamLink": self.streamLink,
            "rois": {
                "cabin1": self.rois[0]
            }
        }
        self.areaName = 'A1/A2'
        self.plant = 'EEL'
        self.init_explosive_parameters()
        self.names_exp = self.model_exp.module.names if hasattr(
            self.model_exp, 'module') else self.model_exp.names
        self.stride_exp = int(self.model_exp.stride.max())
        self.init_explosive_parameters()
        self.names_exp = self.model_exp.module.names if hasattr(
            self.model_exp, 'module') else self.model_exp.names
        self.stride_exp = int(self.model_exp.stride.max())
        self.init_explosive_parameters()
        self.names_exp = self.model_exp.module.names if hasattr(
            self.model_exp, 'module') else self.model_exp.names
        self.stride_exp = int(self.model_exp.stride.max())
        self.init_explosive_parameters()

    def init_explosive_parameters(self):
        """
        Initialize explosive detection parameters.

        This method loads the explosive detection model, retrieves the ROIs from the area data,
        and assigns them to the `rois` attribute.
        """
        self.model_exp = attempt_load(
            'weights/happy_27_05_22.pt', map_location=self.processing_device)
        assert "rois" in self.areaData

        self.rois = list()
        for cabin, roi in self.areaData["rois"].items():
            self.rois.append(roi)

    def start_input(self):
        """
        Start the input stream.

        This method continuously checks if the video capture is opened. If not, it initializes
        the video capture. If it is already opened, it updates the current frame.
        """
        while True:
            if self.capOpened is False:
                self.init_capture()
            else:
                self.update_frame()

    def init_capture(self):
        """
        Initialize the video capture.

        This method initializes the video capture object and sets the `capOpened` flag to True.
        """
        self.cap = cv2.VideoCapture(self.streamLink)
        self.capOpened = True
        _, self.frame = self.cap.read()

    def update_frame(self):
        """
        Update the current frame.

        This method reads the next frame from the video capture, resizes it, converts it to RGB,
        and prepares it for model inference.
        """
        _, self.frame = self.cap.read()
        self.frame = cv2.resize(self.frame, (1280, 720))
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.frame = letterbox(self.frame, new_shape=640)[0]
        self.frame = self.frame[:, :, ::-1].transpose(2, 0, 1)
        self.frame = np.ascontiguousarray(self.frame)
        self.frame = torch.from_numpy(self.frame).to(self.device)
        self.frame = self.frame.float()
        self.frame /= 255.0
        if self.frame.ndimension() == 3:
            self.frame = self.frame.unsqueeze(0)

    def read(self):
        """
        Read the current frame.

        Returns:
            numpy.ndarray: The current frame from the video stream.
        """
        return self.frame

    def run_yolov5s_exp(self, frame_org=None):
        """
        Run the explosive detection model on the frame.

        Args:
            frame_org (numpy.ndarray, optional): The frame to run the model on. If not provided,
                the current frame will be used.

        Returns:
            List: List of predicted bounding boxes, confidence scores, and class labels.
        """
        if frame_org is None:
            frame_org = self.read()

        frame = letterbox(frame_org, 640,
                          stride=self.stride_exp, auto=True)[0]
        # Convert
        frame = frame.transpose((2, 0, 1))[::-1]
        frame = np.ascontiguousarray(frame)
        frame = torch.from_numpy(frame).to(self.processing_device)

        frame = frame / 255.0
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
        pred = self.model_exp(frame, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)
        return pred

    def run_yolov5s(self, frame_org=None):
        """
        Run the object detection model on the frame.

        Args:
            frame_org (numpy.ndarray, optional): The frame to run the model on. If not provided,
                the current frame will be used.

        Returns:
            List: List of predicted bounding boxes, confidence scores, and class labels.
        """
        if frame_org is None:
            frame_org = self.read()

        frame = letterbox(frame_org, 640,
                          stride=self.stride, auto=True)[0]
        # Convert
        frame = frame.transpose((2, 0, 1))[::-1]
        frame = np.ascontiguousarray(frame)
        frame = torch.from_numpy(frame).to(self.processing_device)

        frame = frame / 255.0
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
        pred = self.model(frame, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)
        return pred

    def process(self):
        """
        Process the frame using the object detection model.

        Returns:
            numpy.ndarray: The processed frame with bounding boxes and labels.
        """
        pred = self.run_yolov5s()
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(
                self.frame.shape[2:], det[:, :4], self.frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, self.frame, label=label, color=(0, 255, 0))
        return self.frame

    def process_exp(self):
        """
        Process the frame using the explosive detection model.

        Returns:
            numpy.ndarray: The processed frame with bounding boxes and labels.
        """
        pred = self.run_yolov5s_exp()
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(
                self.frame.shape[2:], det[:, :4], self.frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names_exp[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, self.frame, label=label, color=(0, 255, 0))
        return self.frame

    def process_camera(self):
        """
        Process the frame based on the area type.

        If the area type is 'explosive', the frame is processed using the explosive detection model.
        Otherwise, it is processed using the object detection model.

        Returns:
            numpy.ndarray: The processed frame with bounding boxes and labels.
        """
        if self.areaType == 'explosive':
            return self.process_exp()
        return self.process()
    
   


def main():
    detector = detect_tray()
    while True:
        frame = detector.process_camera()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


        
