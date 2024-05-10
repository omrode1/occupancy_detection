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

