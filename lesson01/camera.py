# -*- coding: utf-8 -*-
import cv2

if __name__ == '__main__':
    #使用するカメラを指定する
    camera = cv2.VideoCapture(0)

    while True:
        #映像を取得する
        ret, frame = camera.read()

        #取得した映像を表示する
        cv2.imshow("camera",frame)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break
