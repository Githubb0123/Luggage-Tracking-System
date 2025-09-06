import cv2
from ultralytics import YOLO
import pyttsx3


engine = pyttsx3.init()
engine.setProperty('rate', 150)

model = YOLO('yolov8x.pt')  


LUGGAGE_LABELS = {'backpack', 'handbag', 'suitcase'}


print("🔍 Choose input source:")
print("1. Live webcam")
print("2. Video file")
choice = input("Enter 1 or 2: ").strip()

if choice == '1':
    source = 0
else:
    source = input("🎞️ Enter video file name (e.g., video.mp4): ").strip()

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("❌ Could not open video source.")
    exit()


announced_luggage = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

  
    results = model.predict(source=frame, conf=0.3, iou=0.5, imgsz=640, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in LUGGAGE_LABELS:
    
            color = (0, 255, 0)
            thickness = 2

            if label not in announced_luggage:
                engine.say(f"{label} detected")
                engine.runAndWait()
                announced_luggage.add(label)
        else:
            color = (0, 0, 255)
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("🎒 Luggage Detector - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
