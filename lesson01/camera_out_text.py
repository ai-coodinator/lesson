# -*- coding: utf-8 -*-
import cv2

if __name__ == '__main__':
    #使用するカメラを指定する
    camera = cv2.VideoCapture(0)

    #出力形式を定義する
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(fps,width,height)

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writer = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    print(width,height,fps)

    while True:
        #映像を取得する
        ret, frame = camera.read()

        #フレームの取得に失敗または動画の末尾
        if not ret:
            break

        #長方形
        cv2.rectangle(frame,(384,0),(510,128),(0,255,0),3)
        #円
        cv2.circle(frame,(447,63), 63, (0,0,255), -1)
        #テキスト
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'OpenCV',(10,300), font, 4,(255,255,255),2,cv2.LINE_AA)

        #動画をフレームごとに書き込む
        writer.write(frame)

        #取得した映像を表示する
        cv2.imshow("camera",frame)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break
