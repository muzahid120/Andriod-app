import kivy
from kivy.core import text
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.button import Button
import cv2 as cv
import mediapipe as mp 

class MyApp(App):
    def build(self):
        # return Button(text='Close',font_size=32)
        facemesh=mp.solutions.face_mesh

        face=facemesh.FaceMesh(static_image_mode=True,min_detection_confidence=0.6,min_tracking_confidence=0.6)
        draw=mp.solutions.drawing_utils
        cap =cv.VideoCapture(0)
        mp.solutions.face_mesh

        while(True):
            rect,frame=cap.read()
            rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            op=face.process(rgb)
            
            #print(dir(op))
            if op.multi_face_landmarks:
                for i in op.multi_face_landmarks:
                    draw.draw_landmarks(frame,i, facemesh.FACEMESH_TESSELATION,landmark_drawing_spec=draw.DrawingSpec(circle_radius=1,color=(0,255,0)))

            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

            
if __name__=='__main__':

    MyApp().run()