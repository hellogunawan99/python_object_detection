from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open a connection to the webcam (usually index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Process the video stream frame by frame
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform object detection on the current frame
        results = model(frame)

        # Display the results
        annotated_frame = results[0].plot()  # Plot results on the frame
        cv2.imshow("YOLO Detection", annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
