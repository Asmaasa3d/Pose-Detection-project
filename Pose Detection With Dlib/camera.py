import cv2
import dlib
import pickle
import numpy as np
from math import cos, sin
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# MinMaxScaler
# StandardScaler
scale= StandardScaler()

filename = 'yaw_finalized_model.sav'
filenameroll = 'random_reg_roll2d.sav'
filenamepitch = 'pitch_model_81_kneigh.sav'



cap = cv2.VideoCapture('mirrrrrrrrra.mp4')

output = "video.avi"


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 20.0, (640,480))

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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




loaded_model_yaw = pickle.load(open(filename, 'rb'))
loaded_model_roll = pickle.load(open(filenameroll, 'rb'))
loaded_model_pitch= pickle.load(open(filenamepitch, 'rb'))
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        # print(face_landmarks)
        
        tst=[]
        
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            tst.append(x)
            tst.append(y)
            cv2.circle(frame, (x, y), 1, (36,255,12), 3)
        tst=np.array(tst).reshape(1,-1)
        scaled_data = scale.fit_transform(tst) 

        # pitch_feat=tst[74,75,76,77,80,81,86,87,88,89,92,93,60,61,62,63,66,67,68,69,70,71,132,133,134,135,16,17,18,19,20,21]
        yhat_yaw=loaded_model_yaw.predict(scaled_data)
        yhat_pitch=loaded_model_pitch.predict(tst)
        yhat_roll=loaded_model_roll.predict(tst)
        
        
        cv2.putText(frame, "Yaw: "+str(yhat_yaw), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
        cv2.putText(frame, "pitch: "+str(yhat_pitch), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
        cv2.putText(frame, "roll: "+str(yhat_roll), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
   
   

    cv2.imshow("Face Landmarks",draw_axis(frame,yhat_yaw,yhat_pitch,yhat_roll
                                 ,tst[:,60],tst[:,61])
            )
    # cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()