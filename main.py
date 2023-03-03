from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
model.fuse()

CLASS_NAME_DICT = model.model.names


video = cv2.VideoCapture(0)

while True:
    rval, frame = video.read()
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy()[0]
            class_id = box.cls.cpu().numpy().astype(int)[0]
            confident = box.conf.cpu().numpy().astype(float)[0]

            cv2.rectangle(img=frame, pt1=(int(x1), int(y1)),
                          pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)
            
            cv2.putText(
                img=frame,
                text=f'{CLASS_NAME_DICT[class_id]} {confident:0.2f}',
                org=(int(x1+5), int(y1-5)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.5, color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )

    cv2.imshow('yolov8', frame)

    key = cv2.waitKey(20)

    if key == 27:
        break

video.release()
cv2.destroyWindow('yolov8')
