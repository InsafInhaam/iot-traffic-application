import cv2
import os
import glob
import shutil

INPUT_FOLDER = "toy_vehicles"
os.makedirs("images", exist_ok=True)
os.makedirs("labels", exist_ok=True)

points = []
boxes = []
classes = []
current_class = 0   # default class = "car"
current_image = None
image_index = 0

class_names = ["car", "bus", "jeep", "truck", "ambulance"]

image_paths = sorted(
    glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(INPUT_FOLDER, "*.png")) +
    glob.glob(os.path.join(INPUT_FOLDER, "*.jpeg"))
)


def click_event(event, x, y, flags, param):
    global points, boxes, classes, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(current_image, (x, y), 4, (0, 255, 255), -1)

        if len(points) > 1:
            cv2.line(current_image, points[-2], points[-1], (0, 255, 255), 2)

        if len(points) == 4:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            boxes.append((x1, y1, x2, y2))
            classes.append(current_class)

            cv2.rectangle(current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(current_image, class_names[current_class], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            points = []


def convert_to_yolo(x1, y1, x2, y2, w, h):
    x_center = (x1 + x2) / 2 / w
    y_center = (y1 + y2) / 2 / h
    width = abs(x2 - x1) / w
    height = abs(y2 - y1) / h
    return x_center, y_center, width, height


cv2.namedWindow("Annotator")
cv2.setMouseCallback("Annotator", click_event)

print("\nControls:")
print("Left Click → 1 point (4 points = box)")
print("Keys 0-4 → Select class")
print("S → Save image")
print("R → Reset")
print("Q → Quit\n")

while image_index < len(image_paths):

    img_path = image_paths[image_index]
    orig = cv2.imread(img_path)
    current_image = orig.copy()
    boxes = []
    points = []
    classes = []

    while True:
        cv2.imshow("Annotator", current_image)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            current_class = int(chr(key))
            print(f"Selected class: {class_names[current_class]}")

        if key == ord('s'):
            h, w, _ = orig.shape
            file_id = f"img_{image_index:04}"

            out_img = f"images/{file_id}.jpg"
            shutil.copy(img_path, out_img)

            out_lbl = f"labels/{file_id}.txt"
            with open(out_lbl, "w") as f:
                for (x1, y1, x2, y2), cls in zip(boxes, classes):
                    xc, yc, ww, hh = convert_to_yolo(x1, y1, x2, y2, w, h)
                    f.write(f"{cls} {xc} {yc} {ww} {hh}\n")

            print(f"Saved {out_img} + {out_lbl}")
            image_index += 1
            break

        if key == ord('r'):
            boxes = []
            points = []
            classes = []
            current_image = orig.copy()
            print("Reset labels")

        if key == ord('q'):
            image_index = len(image_paths)
            break

cv2.destroyAllWindows()
