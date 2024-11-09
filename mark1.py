from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Path to your video file
video_path = "directory/video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Process the video stream frame by frame
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Perform object detection on the current frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = box.cls  # Class label
                confidence = box.conf  # Confidence score

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label and confidence score
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with annotations
        cv2.imshow("YOLO Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the video and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
