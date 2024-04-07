
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
import webbrowser
import platform

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam App")

        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None
        self.canvas = tk.Canvas(window, width=1600, height=900)
        self.canvas.pack()
        self.download_button = tk.Button(window, text="Capture", command=self.download_image)
        self.download_button.pack()
        self.update_webcam()


    def update_webcam(self):
        ret, frame = self.video_capture.read()

        if ret:
        # Flip the frame horizontally (unmirror the image)
            frame = cv2.flip(frame, 1)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(15, self.update_webcam)


    def download_image(self):
        if self.current_image is not None:
            file_path = os.path.expanduser("~/Desktop/Hack 112/HACK112-TEAM_IND/captured_image.jpg")
            self.current_image.save(file_path)
            self.open_file(file_path)
            self.stop_webcam()

    def stop_webcam(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.download_button.config(state="disabled")  # Disable the capture button
            self.canvas.delete("all")  # Optionally clear the canvas

    def open_file(self, file_path):
        if platform.system() == "Darwin":  # macOS
            os.system(f"open \"{file_path}\"")

root = tk.Tk()
app = WebcamApp(root)
root.mainloop()


