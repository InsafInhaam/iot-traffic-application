import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train2/weights/best.pt"
CONFIDENCE = 0.25

# Load model
model = YOLO(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)
print("Classes:", model.names)

# Try multiple camera indexes (macOS fix)
cap = None
for i in range(5):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"🎥 Using camera index {i}")
        break

if cap is None:
    print("❌ No camera found")
    exit()

# Force resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame grab failed — check camera permissions")
        break

    results = model(frame, conf=CONFIDENCE, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("YOLO Toy Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
