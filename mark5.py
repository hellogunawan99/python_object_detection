import cv2
import numpy as np
from scipy.spatial import distance
import torch
from ultralytics import YOLO
import time
from collections import defaultdict


class YOLOTruckDetector:
    def __init__(self, weights_path='directory model', video_source=0):
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
        self.conf_thres = 0.05

        # Parameter untuk tracking kecepatan yang lebih halus
        self.prev_positions = defaultdict(
            lambda: {'positions': [], 'times': []})
        self.truck_speeds = defaultdict(float)
        self.track_history = defaultdict(list)

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

        # Update track history
        self.track_history[truck_id].append(current_pos)
        if len(self.track_history[truck_id]) > self.max_history:
            self.track_history[truck_id].pop(0)

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
        """Hitung jarak antar truk"""
        distances = []

        if not self.pixels_per_meter:
            return distances

        for i, (x1, y1, w1, h1, id1, _) in enumerate(trucks):
            center1 = (x1 + w1//2, y1 + h1//2)

            for j, (x2, y2, w2, h2, id2, _) in enumerate(trucks[i+1:], i+1):
                center2 = (x2 + w2//2, y2 + h2//2)

                pixel_distance = distance.euclidean(center1, center2)
                real_distance = pixel_distance / self.pixels_per_meter

                distances.append((id1, id2, real_distance, center1, center2))

        return distances

    def draw_results(self, frame, trucks, distances):
        """Gambar hasil deteksi dan pengukuran"""
        result_frame = frame.copy()

        if not self.pixels_per_meter:
            return result_frame

        # Gambar track history
        for track_id, history in self.track_history.items():
            if len(history) > 1:
                points = np.hstack(history).astype(
                    np.int32).reshape((-1, 1, 2))
                cv2.polylines(result_frame, [points],
                              False, (230, 230, 230), 1)

        # Gambar box untuk setiap truk
        for x, y, w, h, track_id, speed in trucks:
            # Warna berdasarkan kecepatan
            color = self.get_speed_color(speed)

            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)

            # Tampilkan ID dan kecepatan
            info_text = f'ID: {track_id} Speed: {speed:.1f} km/h'
            cv2.putText(result_frame, info_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Tambahkan dimensi truk
            dimensions_text = '{:.1f}m x {:.1f}m'.format(
                w/self.pixels_per_meter,
                h/self.pixels_per_meter
            )
            cv2.putText(result_frame, dimensions_text, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Gambar garis dan jarak antar truk
        for truck1_id, truck2_id, dist, center1, center2 in distances:
            cv2.line(result_frame, center1, center2, (255, 0, 0), 2)
            mid_point = ((center1[0] + center2[0])//2,
                         (center1[1] + center2[1])//2)
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
        weights_path='directory model',
        video_source='directory video'
    )

    # Run detection
    detector.run()


if __name__ == "__main__":
    main()
