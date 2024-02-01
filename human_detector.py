import cv2
import pandas as pd
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if(event == cv2.EVENT_MOUSEMOVE):
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture(0)

output = cv2.VideoWriter('output_final.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (1020, 500))

file = open(r'C:\Users\nemob\Downloads\PeopleCounting-ComputerVision-master\PeopleCounting-ComputerVision-master\coco.names', 'r')
data = file.read()
class_list = data.split('\n')

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if (count % 3 != 0):
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    num_persons = 0

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        if 'person' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            num_persons += 1

    cv2.putText(frame, f'Number of Persons: {num_persons}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('RGB', frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()