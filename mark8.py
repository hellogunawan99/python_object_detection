import cv2
import numpy as np
from scipy.spatial import distance
import torch
from ultralytics import YOLO
import time
from collections import defaultdict


class YOLOTruckDetector:
    def __init__(self, weights_path='/Users/gunawan/dev/pythonapp/computer_vision/truck_detection/yolo_v11/bestv3.pt', video_source=0):
        # Inisialisasi YOLO model
        print("Loading YOLOv11-S model...")
        self.model = YOLO(weights_path)

        # Inisialisasi video capture
        print("Starting video capture...")
        self.cap = cv2.VideoCapture(video_source)

        # Set resolusi kamera (opsional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Dimensi truk tambang (dalam meter)
        self.truck_length = 11.0
        self.truck_height = 5.0
        self.truck_width = 4.95

        # Inisialisasi pixels_per_meter
        self.pixels_per_meter = None

        # Confidence threshold untuk deteksi
        self.conf_thres = 0.4

        # Threshold jarak (dalam meter)
        self.distance_threshold = 100.0

        # Parameter untuk tracking kecepatan yang lebih halus
        self.prev_positions = defaultdict(
            lambda: {'positions': [], 'times': []})
        self.truck_speeds = defaultdict(float)

        # Hapus track_history karena tidak digunakan lagi
        # self.track_history = defaultdict(list)

        # Parameter untuk smoothing kecepatan
        self.speed_smoothing_factor = 0.1  # Nilai lebih kecil = lebih smooth
        self.position_history_size = 5  # Jumlah posisi untuk rata-rata
        self.speed_update_interval = 2.0  # Update speed setiap 1 detik
        self.max_history = 10
        self.last_speed_update = defaultdict(float)

        # Filter untuk mencegah perubahan kecepatan yang terlalu drastis
        self.max_speed_change = 5.0  # km/h per detik
        self.min_speed = 0.0
        self.max_speed = 60.0

    def enhance_night_image(self, frame):
        """Meningkatkan visibility gambar malam"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def calculate_speed(self, truck_id, current_pos, current_time):
        """Hitung kecepatan objek dalam km/h dengan smoothing yang lebih baik"""
        if not self.pixels_per_meter:
            return 0.0

        prev_data = self.prev_positions[truck_id]

        # Tambahkan posisi dan waktu baru ke history
        prev_data['positions'].append(current_pos)
        prev_data['times'].append(current_time)

        # Batasi ukuran history
        if len(prev_data['positions']) > self.position_history_size:
            prev_data['positions'].pop(0)
            prev_data['times'].pop(0)

        # Cek apakah sudah waktunya update kecepatan
        if truck_id not in self.last_speed_update:
            self.last_speed_update[truck_id] = current_time

        time_since_last_update = current_time - \
            self.last_speed_update[truck_id]

        # Hanya update kecepatan setelah interval tertentu
        if time_since_last_update < self.speed_update_interval:
            return self.truck_speeds[truck_id]

        # Hitung kecepatan hanya jika memiliki cukup history
        if len(prev_data['positions']) >= 2:
            # Gunakan posisi rata-rata untuk mengurangi noise
            start_pos = np.mean(prev_data['positions'][:2], axis=0)
            end_pos = np.mean(prev_data['positions'][-2:], axis=0)

            # Hitung jarak dalam meter
            pixel_distance = distance.euclidean(start_pos, end_pos)
            distance_meters = pixel_distance / self.pixels_per_meter

            # Hitung waktu dalam detik
            time_diff = prev_data['times'][-1] - prev_data['times'][0]

            if time_diff > 0:
                # Hitung kecepatan dalam km/h
                speed_ms = distance_meters / time_diff
                new_speed = speed_ms * 3.6  # Convert m/s to km/h

                # Terapkan batas perubahan kecepatan
                if truck_id in self.truck_speeds:
                    prev_speed = self.truck_speeds[truck_id]
                    max_change = self.max_speed_change * time_since_last_update
                    speed_change = new_speed - prev_speed

                    if abs(speed_change) > max_change:
                        new_speed = prev_speed + \
                            (max_change if speed_change > 0 else -max_change)

                # Terapkan batas minimum dan maximum
                new_speed = np.clip(new_speed, self.min_speed, self.max_speed)

                # Aplikasikan smoothing
                if truck_id in self.truck_speeds:
                    new_speed = (self.speed_smoothing_factor * new_speed +
                                 (1 - self.speed_smoothing_factor) * self.truck_speeds[truck_id])

                self.truck_speeds[truck_id] = new_speed
                self.last_speed_update[truck_id] = current_time

        return self.truck_speeds[truck_id]

    def detect_trucks(self, frame):
        """Deteksi truk menggunakan YOLO"""
        # Tingkatkan kualitas gambar malam
        enhanced_frame = self.enhance_night_image(frame)

        # Inference dengan YOLO
        results = self.model(
            enhanced_frame, conf=self.conf_thres, verbose=False)

        # Ekstrak bounding boxes dan tracking IDs
        trucks = []
        current_time = time.time()

        if hasattr(results[0], 'boxes'):
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result[:6]
                track_id = int(result[6]) if len(result) > 6 else len(
                    trucks)  # Use detection index if no tracking ID

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                center = (x + w//2, y + h//2)
                speed = self.calculate_speed(track_id, center, current_time)

                trucks.append((x, y, w, h, track_id, speed))

        # Hitung pixels_per_meter
        if trucks:
            lengths_in_pixels = [w for _, _, w, h, _, _ in trucks]
            avg_length_pixels = np.mean(lengths_in_pixels)
            self.pixels_per_meter = avg_length_pixels / self.truck_length

        return trucks

    def calculate_distances(self, trucks):
        """Calculate distances between trucks with improved accuracy"""
        distances = []

        if not self.pixels_per_meter:
            return distances

        for i, (x1, y1, w1, h1, id1, _) in enumerate(trucks):
            # Calculate corners of first truck
            corners1 = self.get_truck_corners(x1, y1, w1, h1)

            for j, (x2, y2, w2, h2, id2, _) in enumerate(trucks[i+1:], i+1):
                # Calculate corners of second truck
                corners2 = self.get_truck_corners(x2, y2, w2, h2)

                # Find minimum distance between any two points on the trucks
                min_distance = float('inf')
                closest_points = None

                for p1 in corners1:
                    for p2 in corners2:
                        dist = distance.euclidean(p1, p2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_points = (p1, p2)

                # Convert pixel distance to meters
                real_distance = min_distance / self.pixels_per_meter

                # Apply perspective correction
                real_distance = self.correct_perspective(real_distance,
                                                         closest_points[0][1],
                                                         closest_points[1][1])

                # Only add if distance is within threshold
                if real_distance < self.distance_threshold:
                    distances.append(
                        (id1, id2, real_distance, closest_points[0], closest_points[1]))

        return distances

    def get_truck_corners(self, x, y, w, h):
        """Get the four corners of a truck bounding box"""
        return [
            (x, y),           # Top-left
            (x + w, y),       # Top-right
            (x, y + h),       # Bottom-left
            (x + w, y + h)    # Bottom-right
        ]

    def correct_perspective(self, distance, y1, y2):
        """
        Apply perspective correction based on vertical position in image
        Objects further up in image (smaller y) appear closer than they are
        """
        # Get image height
        frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Calculate correction factors based on vertical position
        # Objects at the top of frame need more correction than bottom
        correction1 = 1 + (1 - y1/frame_height) * 0.3  # 30% max correction
        correction2 = 1 + (1 - y2/frame_height) * 0.3

        # Apply average correction factor
        avg_correction = (correction1 + correction2) / 2

        return distance * avg_correction

    def draw_results(self, frame, trucks, distances):
        """Draw detection results with improved visualization"""
        result_frame = frame.copy()

        if not self.pixels_per_meter:
            return result_frame

        # Draw trucks
        for x, y, w, h, track_id, speed in trucks:
            color = self.get_speed_color(speed)

            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)

            # Draw corners for visibility
            corners = self.get_truck_corners(x, y, w, h)
            for corner in corners:
                cv2.circle(result_frame, corner, 3, color, -1)

            # Draw info text
            info_text = f'ID: {track_id} Speed: {speed:.1f} km/h'
            cv2.putText(result_frame, info_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            dimensions_text = '{:.1f}m x {:.1f}m'.format(
                w/self.pixels_per_meter,
                h/self.pixels_per_meter
            )
            cv2.putText(result_frame, dimensions_text, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw distances
        for truck1_id, truck2_id, dist, p1, p2 in distances:
            line_color = self.get_distance_color(dist)

            # Draw line between closest points
            cv2.line(result_frame, p1, p2, line_color, 2)

            # Draw distance label
            mid_point = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
            cv2.putText(result_frame, f'{dist:.2f}m', mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return result_frame

    def get_speed_color(self, speed):
        """Return color based on speed (BGR format)"""
        if speed < 10:  # Slow
            return (0, 255, 0)  # Green
        elif speed < 30:  # Medium
            return (0, 255, 255)  # Yellow
        else:  # Fast
            return (0, 0, 255)  # Red

    def get_distance_color(self, distance):
        """Return color based on distance (BGR format)"""
        if distance < 20:  # Very close
            return (0, 0, 255)  # Red
        elif distance < 35:  # Medium distance
            return (0, 165, 255)  # Orange
        else:  # Far but still under threshold
            return (0, 255, 255)  # Yellow

    def run(self):
        """Jalankan deteksi secara real-time"""
        print("Detection started. Press 'q' to quit.")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error membaca frame dari kamera")
                    break

                # Deteksi truk menggunakan YOLO
                trucks = self.detect_trucks(frame)

                # Hitung jarak
                distances = self.calculate_distances(trucks)

                # Gambar hasil
                result_frame = self.draw_results(frame, trucks, distances)

                # Tampilkan FPS
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Tampilkan frame
                cv2.imshow('YOLO Mining Truck Detection', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __del__(self):
        """Destructor untuk memastikan kamera dilepas"""
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    # Initialize detector with YOLO weights and video source
    detector = YOLOTruckDetector(
        weights_path='/Users/gunawan/dev/pythonapp/computer_vision/truck_detection/yolo_v11/bestv3.pt',
        video_source='/Users/gunawan/dev/pythonapp/computer_vision/truck_detection/yolo_v11/1109.mp4'
    )

    # Run detection
    detector.run()


if __name__ == "__main__":
    main()
