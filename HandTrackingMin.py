import time
import cv2
import HandTrackingModule as ht


def main():
    p_time = 0
    cap = cv2.VideoCapture(1)
    detector = ht.HandDetector()  # Adjusted track_con to be an integer

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if lm_list:
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("image", img)
        cv2.waitKey(2)


if __name__ == '__main__':
    main()
