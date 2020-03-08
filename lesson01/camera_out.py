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

    # fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    # writer = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    while True:
        #映像を取得する
        ret, frame = camera.read()

        #フレームの取得に失敗または動画の末尾
        if not ret:
            break

        #動画をフレームごとに書き込む
        writer.write(frame)

        #取得した映像を表示する
        cv2.imshow("camera",frame)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break
