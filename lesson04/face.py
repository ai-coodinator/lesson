# -*- coding: utf-8 -*-
import cv2

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

if __name__ == '__main__':
    #使用するカメラを指定する
    camera = cv2.VideoCapture(0)

    while True:
        #映像を取得する
        ret, frame = camera.read()

        cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
        face_list = cascade.detectMultiScale(frame,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(100,100))

        if len(face_list) > 0:
            # 認識した部分を赤色で囲む
            print(face_list)

            color = (0, 0, 255)
            for face in face_list:
                x,y,w,h = face
                frame = mosaic_area(frame, x,y,w,h)
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, thickness=3)


        #取得した映像を表示する
        cv2.imshow("camera",frame)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break
