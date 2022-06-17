import cv2
import pickle
import numpy as np
from math import cos, sin
from sklearn.preprocessing import StandardScaler
import mediapipe

filename = 'yaw_svr_standarascalar.sav'
filenameroll ='roll_svr_standarascalar.sav'
filenamepitch ='pitch_svr_standarascalar.sav'

cap = cv2.VideoCapture('asmaa22.mp4')
# cap = cv2.VideoCapture(0)

# cv2.namedWindow("output", cv2.WINDOW_NORMAL)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),20,(640,480))

loaded_model_yaw = pickle.load(open(filename, 'rb'))
loaded_model_roll = pickle.load(open(filenameroll, 'rb'))
loaded_model_pitch= pickle.load(open(filenamepitch, 'rb'))
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faceModule = mediapipe.solutions.face_mesh
    shape = frame.shape 
    # loading image and its correspinding mat file
    with faceModule.FaceMesh(static_image_mode=True) as faces:
        results = faces.process(frame)
        if results.multi_face_landmarks != None: 
          # looping over the faces in the image
          for face in results.multi_face_landmarks:
              tst=[]
              for landmark in face.landmark:
                  x = landmark.x
                  y = landmark.y
                  # note: the x and y values are scaled to the their width and height so we will get back their actual value in the image
                  shape = frame.shape 
                  relative_x = int(x * shape[1])
                  relative_y = int(y * shape[0])
                  tst.append(x)
                  tst.append(y)
                  # cv2.putText(image, str(relative_y), (int(relative_x),int(relative_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
                  cv2.circle(frame, (relative_x, relative_y), radius=1, color=(0, 255, 0), thickness=2)
              tst=np.array(tst).reshape(1,-1)
              tst1=tst
              for i in range(0, len(tst)):
                    if i % 2:
                        tst[i,:]-=tst[i,8]
                    else :
                        tst[i,:]-=tst[i,9]
              
            #   scale= StandardScaler()
            #   scaled_data = scale.fit_transform(tst) 
              yhat_yaw=loaded_model_yaw.predict(tst)
              yhat_pitch=loaded_model_pitch.predict(tst)
              yhat_roll=loaded_model_roll.predict(tst)
              cv2.putText(frame, "Yaw: "+str(yhat_yaw), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
              cv2.putText(frame, "pitch: "+str(yhat_pitch), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
              cv2.putText(frame, "roll: "+str(yhat_roll), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
            #   x = int(tst[:,115*2] * shape[1])
            #   y = int(tst[:,116*2] * shape[0])
              
             #   x_nose=results.multi_face_landmarks.landmark[faceModule.PoseLandmark.NOSE].x
              img=draw_axis(frame,yhat_yaw,yhat_pitch,yhat_roll)
              cv2.imshow("Face Landmarks",img )
              out.write(img)
            #   break
            # cv2.imshow("Face Landmarks", frame)
        # imS = cv2.resize(frame, (500, 500))
        # cv2.imshow("Face Landmarks", imS)
        

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
