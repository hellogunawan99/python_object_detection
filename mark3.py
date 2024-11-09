import cv2
import numpy as np
from scipy.spatial import distance
import torch
from ultralytics import YOLO


class YOLOTruckDetector:
    def __init__(self, weights_path='directory/model.pt', video_source=0):
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
        self.conf_thres = 0.10

    def enhance_night_image(self, frame):
        """Meningkatkan visibility gambar malam"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def detect_trucks(self, frame):
        """Deteksi truk menggunakan YOLO"""
        # Tingkatkan kualitas gambar malam
        enhanced_frame = self.enhance_night_image(frame)

        # Inference dengan YOLO
        results = self.model(enhanced_frame, conf=self.conf_thres)

        # Ekstrak bounding boxes
        trucks = []
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            if conf > self.conf_thres:
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                trucks.append((x, y, w, h))

        # Hitung pixels_per_meter
        if trucks:
            lengths_in_pixels = [w for _, _, w, _ in trucks]
            avg_length_pixels = np.mean(lengths_in_pixels)
            self.pixels_per_meter = avg_length_pixels / self.truck_length

        return trucks

    def calculate_distances(self, trucks):
        """Hitung jarak antar truk"""
        distances = []

        if not self.pixels_per_meter:
            return distances

        for i, (x1, y1, w1, h1) in enumerate(trucks):
            center1 = (x1 + w1//2, y1 + h1//2)

            for j, (x2, y2, w2, h2) in enumerate(trucks[i+1:], i+1):
                center2 = (x2 + w2//2, y2 + h2//2)

                pixel_distance = distance.euclidean(center1, center2)
                real_distance = pixel_distance / self.pixels_per_meter

                distances.append((i, j, real_distance, center1, center2))

        return distances

    def draw_results(self, frame, trucks, distances):
        """Gambar hasil deteksi dan pengukuran"""
        result_frame = frame.copy()

        if not self.pixels_per_meter:
            return result_frame

        # Gambar box untuk setiap truk
        for i, (x, y, w, h) in enumerate(trucks):
            cv2.rectangle(result_frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.putText(result_frame, f'Truck {i}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tambahkan dimensi truk
            dimensions_text = '{:.1f}m x {:.1f}m'.format(
                w/self.pixels_per_meter,
                h/self.pixels_per_meter
            )
            cv2.putText(result_frame, dimensions_text, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Gambar garis dan jarak antar truk
        for truck1_id, truck2_id, dist, center1, center2 in distances:
            cv2.line(result_frame, center1, center2, (255, 0, 0), 2)
            mid_point = ((center1[0] + center2[0])//2,
                         (center1[1] + center2[1])//2)
            cv2.putText(result_frame, f'{dist:.2f}m', mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return result_frame

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
        weights_path='directory/model.pt',
        video_source='directory/video.mp4'
    )

    # Run detection
    detector.run()


if __name__ == "__main__":
    main()
