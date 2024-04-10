import numpy as np
import time

# Define the function to perform object detection on the frame
def detect_objects(frame):
    # Perform object detection using a pre-trained model like YOLO or SSD
    # Return the detections as a list of bounding boxes or as a list of detected objects

# Define the function to calculate the intersection over union (IoU) between two bounding boxes
 def calculate_iou(box1, box2):
    # Calculate the intersection area
    intersection_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    
    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area
    
    pass  # Placeholder to satisfy the expected indented block
    
    return iou

# Initialize variables
occupancy = 0
in_time = {}
out_time = {}

while True:
    # Read the current frame from each video stream
    detections = []  # Define the variable "detections"
    for video in video_streams:
        ret, frame = video.read()
        # Perform object detection on the frame to detect workers
        detected_objects = detect_objects(frame)
        # Append the detections to the "detections" list
        detections.append(detected_objects)

    # Calculate the number of workers in each region of interest
    num_workers = [len(detections_roi) for detections_roi in detections]

    # Update the occupancy count
    occupancy = sum(num_workers)

    # Check if any workers have entered or exited each region of interest
    for roi_index, detections_roi in enumerate(detections):
        for worker_id, worker_bbox in detections_roi.items():
            if worker_id not in in_time:
                in_time[worker_id] = time.time()
            else:
                out_time[worker_id] = time.time()
                # Calculate the time spent by the worker in the region of interest
                time_spent = out_time[worker_id] - in_time[worker_id]
                print(f"Worker {worker_id} in ROI {roi_index}: Time spent - {time_spent}")
                del in_time[worker_id]

    # Display the frame with bounding boxes around the workers
    for detections_roi in detections:
        for worker_bbox in detections_roi.values():
            cv2.rectangle(frame, (worker_bbox[0], worker_bbox[1]), (worker_bbox[2], worker_bbox[3]), (0, 255, 0), 2)
    cv2.imshow("Occupancy Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video or camera feeds
for video in video_streams:
    video.release()
cv2.destroyAllWindows()