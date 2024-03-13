import kivy
from kivy.graphics.texture import Texture

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
import mediapipe as mp

class MyApp(App):
    def build(self):
        self.facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)  # Update at 30 fps

        layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        layout.add_widget(self.img1)

        quit_button = Button(text='Quit', size_hint=(1, 0.1))
        quit_button.bind(on_press=self.stop)
        layout.add_widget(quit_button)

        return layout

    def update(self, dt):
        rect, frame = self.cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        op = self.facemesh.process(rgb)

        if op.multi_face_landmarks:
            for face_landmarks in op.multi_face_landmarks:
                self.draw.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                         landmark_drawing_spec=self.draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img1.texture = texture1

    def on_stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    MyApp().run()
