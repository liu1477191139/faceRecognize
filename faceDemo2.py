import cv2

# 读取视频
cap = cv2.VideoCapture("imgs/video01.mp4")

#加载人脸模型库
face_cascade = cv2.CascadeClassifier("plugins/opencv/haarcascade_frontalface_default.xml")
face_cascade.load("plugins/opencv/haarcascade_frontalface_default.xml")

while cap.isOpened():
    ret, frame = cap.read()

    # 图片进行灰度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 标记人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 设置可以收缩
    cv2.namedWindow("frame", 0)

    # 设置窗口大小
    cv2.resizeWindow("frame", 1280, 720)

    # 显示
    cv2.imshow('frame', frame)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()