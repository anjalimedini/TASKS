import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--WIDTH",type=int,help="enter width value")
parser.add_argument("--HEIGHT",type=int,help="enter height value")
parser.add_argument("--SIZE: WIDTH,HEIGHT")
args = parser.parse_args()
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('web_video1.avi', fourcc, 20.0, size)

while(True):
    _, frame = cap.read()
    cv2.imshow('Recording...', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()