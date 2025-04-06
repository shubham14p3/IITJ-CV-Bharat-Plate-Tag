# src/sort.py
# SORT tracker implementation
import numpy as np
from filterpy.kalman import KalmanFilter

class Tracker:
    def __init__(self):
        self.trackers = []
        self.track_id = 0

    def update(self, detections):
        updated_boxes = []
        for det in detections:
            updated_boxes.append(det[:4])
        return updated_boxes

# Simplified dummy SORT logic — Replace with full SORT logic or use a real implementation if needed
