from ultralytics import YOLO
import cv2

# Load YOLO model (use yolov8n for speed)
model = YOLO("yolov8n.pt")   # or a custom trained model for toy vehicles

# Open webcam or video
cap = cv2.VideoCapture(0)  # 0 = webcam, or "video.mp4"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Toy Vehicle Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
