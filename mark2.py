import cv2
import torch
from ultralytics import YOLO


def detect_live(weights_path='directory/model.pt', conf_thres=0.10):
    # Load model
    print("Loading YOLOv11-S model...")
    model = YOLO(weights_path)

    # Buka webcam
    print("Starting video capture...")
    # gunakan 0 untuk webcam default
    cap = cv2.VideoCapture(
        'directory/video.mp4')

    print("Detection started. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # Inference
        results = model(frame, conf=conf_thres)

        # Visualisasi hasil
        annotated_frame = results[0].plot()

        # Tampilkan FPS
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cv2.putText(annotated_frame,
                    f'FPS: {fps}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Tampilkan frame
        cv2.imshow('YOLOv11-S Detection', annotated_frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_live()
