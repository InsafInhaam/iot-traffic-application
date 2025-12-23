import cv2
import requests
import base64
import json

API_KEY = "3ClCXDlvIYLP3qaTeQia"
WORKSPACE = "ai-uydbt"
WORKFLOW = "find-stripes-windows-cars-mirrors-jeeps-buses-trucks-and-ambulances"

URL = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW}"


def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame → base64 for API
    image_b64 = frame_to_base64(frame)

    # Build request
    payload = {
        "api_key": API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_b64
            }
        }
    }

    # Send POST request
response = requests.post(URL, json=payload)
data = response.json()

# ---- FIXED PART HERE ----
outputs = data["outputs"]          # list
img_output = outputs[0]["image"]   # first output
preds = img_output["objects"]      # bounding boxes
# ---------------------------

# Draw bounding boxes
for p in preds:
    box = p["bounding_box"]
    x, y = box["x"], box["y"]
    w, h = box["width"], box["height"]
    cls = p["class"]

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, cls, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Toy Cars via Roboflow Workflow", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
