import cv2

# 读取视频
cap = cv2.VideoCapture("imgs/video01.mp4")

#加载人脸模型库，可以直接把分类器放到CascadeClassofier的参数里，就不需要load
face_cascade = cv2.CascadeClassifier()
face_cascade.load("plugins/opencv/haarcascade_frontalface_default.xml")

while cap.isOpened():

    # 第一个参数用于表示是否抓取了帧
    ret, frame = cap.read()

    # 利用颜色空间转换函数，对图片进行灰度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测，返回一个由列表组成的列表
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 标记人脸
    for (x, y, w, h) in faces:

        # 矩形函数1.原始图片；2坐标点；3.矩形宽高 4.颜色值(RGB)；5.线框
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

# 释放内存
cap.release()

# 销毁窗口资源
cv2.destroyAllWindows()