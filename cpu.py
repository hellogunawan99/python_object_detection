import cv2
import numpy as np
from scipy.spatial import distance
import torch
from ultralytics import YOLO
import time
from collections import defaultdict


class YOLOTruckDetector:
    def __init__(self, weights_path, video_source=0):
        # Initialize YOLO model
        print("Loading YOLOv11-S model...")
        self.device = 'cpu'  # Force CPU usage
        print(f"Device selected: {self.device}")
        
        # Load model
        print(f"Loading model from {weights_path}...")
        self.model = YOLO(weights_path).to(self.device)
        self.model.conf = 0.5  # Confidence threshold
        self.model.iou = 0.5   # IOU threshold
        print("Model loaded successfully!")
        
        # Video capture initialization
        print("Starting video capture...")
        self.cap = cv2.VideoCapture(video_source)
        
        # Get original video resolution
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Processing resolution for YOLO (maintain aspect ratio)
        self.processing_width = 640  # Reduced for CPU processing
        self.scale_factor = self.processing_width / self.original_width
        self.processing_height = int(self.original_height * self.scale_factor)

        # Truck dimensions (meters)
        self.truck_length = 11.0
        self.truck_height = 5.0
        self.truck_width = 4.95

        self.pixels_per_meter = None
        self.conf_thres = 0.5
        self.distance_threshold = 100.0

        # Speed tracking parameters
        self.prev_positions = defaultdict(lambda: {'positions': [], 'times': []})
        self.truck_speeds = defaultdict(float)
        
        # Smoothing parameters with adjusted values for CPU
        self.speed_smoothing_factor = 0.15  # Increased for smoother updates
        self.position_history_size = 3  # Reduced for CPU
        self.speed_update_interval = 2.5  # Increased update interval
        self.max_history = 8  # Reduced history size
        self.last_speed_update = defaultdict(float)

        # Speed limits and filters
        self.max_speed_change = 5.0  # km/h per second
        self.min_speed = 0.0
        self.max_speed = 60.0

        # Frame processing for CPU optimization
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0

    def enhance_night_image(self, frame):
        avg_brightness = np.mean(frame)
        if avg_brightness > 100:  # Skip if bright enough
            return frame
            
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def calculate_speed(self, truck_id, current_pos, current_time):
        if not self.pixels_per_meter:
            return 0.0

        prev_data = self.prev_positions[truck_id]
        prev_data['positions'].append(current_pos)
        prev_data['times'].append(current_time)

        if len(prev_data['positions']) > self.position_history_size:
            prev_data['positions'].pop(0)
            prev_data['times'].pop(0)

        if truck_id not in self.last_speed_update:
            self.last_speed_update[truck_id] = current_time

        time_since_last_update = current_time - self.last_speed_update[truck_id]

        if time_since_last_update < self.speed_update_interval:
            return self.truck_speeds[truck_id]

        if len(prev_data['positions']) >= 2:
            start_pos = np.mean(prev_data['positions'][:2], axis=0)
            end_pos = np.mean(prev_data['positions'][-2:], axis=0)
            pixel_distance = distance.euclidean(start_pos, end_pos)
            distance_meters = pixel_distance / self.pixels_per_meter
            time_diff = prev_data['times'][-1] - prev_data['times'][0]

            if time_diff > 0:
                speed_ms = distance_meters / time_diff
                new_speed = speed_ms * 3.6

                if truck_id in self.truck_speeds:
                    prev_speed = self.truck_speeds[truck_id]
                    max_change = self.max_speed_change * time_since_last_update
                    speed_change = new_speed - prev_speed
                    if abs(speed_change) > max_change:
                        new_speed = prev_speed + (max_change if speed_change > 0 else -max_change)

                new_speed = np.clip(new_speed, self.min_speed, self.max_speed)

                if truck_id in self.truck_speeds:
                    new_speed = (self.speed_smoothing_factor * new_speed +
                              (1 - self.speed_smoothing_factor) * self.truck_speeds[truck_id])

                self.truck_speeds[truck_id] = new_speed
                self.last_speed_update[truck_id] = current_time

        return self.truck_speeds[truck_id]

    def detect_trucks(self, frame):
        # Resize frame for processing while maintaining aspect ratio
        processing_frame = cv2.resize(frame, (self.processing_width, self.processing_height))
        
        # Enhance if dark
        avg_brightness = np.mean(processing_frame)
        enhanced_frame = self.enhance_night_image(processing_frame) if avg_brightness < 100 else processing_frame
        
        # CPU optimization: use float32 instead of half precision
        with torch.no_grad():
            results = self.model(enhanced_frame, conf=self.conf_thres, verbose=False)

        trucks = []
        current_time = time.time()

        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes_data = results[0].boxes.data.cpu().numpy()
            
            for result in boxes_data:
                # Scale coordinates back to original frame size
                x1, y1, x2, y2 = map(float, result[:4])
                x1 = int(x1 / self.scale_factor)
                y1 = int(y1 / self.scale_factor)
                x2 = int(x2 / self.scale_factor)
                y2 = int(y2 / self.scale_factor)
                
                conf = result[4]
                if conf < self.conf_thres:
                    continue
                    
                track_id = int(result[6]) if len(result) > 6 else len(trucks)
                w = x2 - x1
                h = y2 - y1
                center = (x1 + w//2, y1 + h//2)
                speed = self.calculate_speed(track_id, center, current_time)
                trucks.append((x1, y1, w, h, track_id, speed))

            if trucks:
                widths = np.array([w for _, _, w, h, _, _ in trucks])
                self.pixels_per_meter = np.mean(widths) / self.truck_length

        return trucks

    def calculate_distances(self, trucks):
        distances = []
        if len(trucks) < 2 or not self.pixels_per_meter:
            return distances

        for i, (x1, y1, w1, h1, id1, _) in enumerate(trucks):
            center1 = (x1 + w1//2, y1 + h1//2)
            
            for j, (x2, y2, w2, h2, id2, _) in enumerate(trucks[i+1:], i+1):
                center2 = (x2 + w2//2, y2 + h2//2)
                
                approx_dist = distance.euclidean(center1, center2)
                approx_meters = approx_dist / self.pixels_per_meter
                
                if approx_meters > self.distance_threshold * 1.5:
                    continue
                
                corners1 = [(x1, y1), (x1 + w1, y1 + h1)]
                corners2 = [(x2, y2), (x2 + w2, y2 + h2)]
                
                min_distance = float('inf')
                closest_points = None
                
                for p1 in corners1:
                    for p2 in corners2:
                        dist = distance.euclidean(p1, p2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_points = (p1, p2)
                
                real_distance = min_distance / self.pixels_per_meter
                real_distance = self.correct_perspective(
                    real_distance,
                    closest_points[0][1],
                    closest_points[1][1]
                )
                
                if real_distance < self.distance_threshold:
                    distances.append((id1, id2, real_distance, closest_points[0], closest_points[1]))
        
        return distances

    def correct_perspective(self, distance, y1, y2):
        frame_height = self.original_height
        correction1 = 1 + (1 - y1/frame_height) * 0.3
        correction2 = 1 + (1 - y2/frame_height) * 0.3
        avg_correction = (correction1 + correction2) / 2
        return distance * avg_correction

    def draw_results(self, frame, trucks, distances):
        if not trucks or not self.pixels_per_meter:
            return frame
        result_frame = frame.copy()

        for x, y, w, h, track_id, speed in trucks:
            color = self.get_speed_color(speed)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            info_text = f'ID: {track_id} Speed: {speed:.1f} km/h'
            cv2.putText(result_frame, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            dimensions_text = '{:.1f}m x {:.1f}m'.format(
                w/self.pixels_per_meter,
                h/self.pixels_per_meter
            )
            cv2.putText(result_frame, dimensions_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for truck1_id, truck2_id, dist, p1, p2 in distances:
            line_color = self.get_distance_color(dist)
            cv2.line(result_frame, p1, p2, line_color, 2)
            
            mid_point = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
            cv2.putText(result_frame, f'{dist:.2f}m', mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return result_frame

    def get_speed_color(self, speed):
        if speed < 10:
            return (0, 255, 0)
        elif speed < 30:
            return (0, 255, 255)
        else:
            return (0, 0, 255)

    def get_distance_color(self, distance):
        if distance < 20:
            return (0, 0, 255)
        elif distance < 35:
            return (0, 165, 255)
        else:
            return (0, 255, 255)

    def run(self):
        print("Detection started. Press 'q' to quit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                # CPU optimization: skip frames
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                trucks = self.detect_trucks(frame)
                distances = self.calculate_distances(trucks)
                result_frame = self.draw_results(frame, trucks, distances)

                # Create window with full screen
                cv2.namedWindow('YOLO Mining Truck Detection', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('YOLO Mining Truck Detection', cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_FULLSCREEN)

                # Calculate actual FPS
                fps = self.cap.get(cv2.CAP_PROP_FPS) / self.frame_skip
                cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('YOLO Mining Truck Detection', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    detector = YOLOTruckDetector(
        weights_path='C:\\Users\\Buanamahati\\dev\\computer_vision\\python_object_detection\\bestv3.pt',
        video_source='C:\\Users\\Buanamahati\\dev\\computer_vision\\python_object_detection\\1109.mp4'
    )
    detector.run()


if __name__ == "__main__":
    main()