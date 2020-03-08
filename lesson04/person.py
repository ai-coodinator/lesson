# -*- coding: utf-8 -*-
import cv2

if __name__ == '__main__':

    camera = cv2.VideoCapture("person.mp4")

    while True:
        #映像を取得する
        ret, frame = camera.read()

        cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_fullbody.xml")
        face_list = cascade.detectMultiScale(frame,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(100,100))

        if len(face_list) > 0:
            # 認識した部分を赤色で囲む
            # print(face_list)
            color = (0, 0, 255)
            for face in face_list:
                x,y,w,h = face
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, thickness=3)

        #取得した映像を表示する
        cv2.imshow("camera",frame)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break
