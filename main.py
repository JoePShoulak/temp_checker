import cv2
import pytesseract
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class TempMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Temp Monitor (Single ROI with Threshold Slider)")

        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.screen_roi = None
        self.zoomed_roi = None

        self.canvas = Canvas(root, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=2)

        Label(root, text="Tag:").grid(row=1, column=0)
        self.tag = StringVar()
        Entry(root, textvariable=self.tag).grid(row=1, column=1)

        self.label = Label(root, text="Reading: -- °C", font=('Arial', 14))
        self.label.grid(row=2, column=0, columnspan=2)

        # Threshold slider
        Label(root, text="Threshold:").grid(row=3, column=0)
        self.threshold_value = IntVar(value=80)
        Scale(root, from_=80, to=120, orient=HORIZONTAL, variable=self.threshold_value).grid(row=3, column=1)

        self.drawing_screen = False
        self.screen_start = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self.start_screen_roi)
        self.canvas.bind("<B1-Motion>", self.draw_screen_roi_motion)
        self.canvas.bind("<ButtonRelease-1>", self.finish_screen_roi)

        # Zoomed-in view
        self.zoom_window = Toplevel(root)
        self.zoom_window.title("Zoomed Screen (Select ROI)")
        self.zoom_canvas = Canvas(self.zoom_window, width=640, height=480)
        self.zoom_canvas.pack()

        self.drawing_zoom = False
        self.zoom_start = (0, 0)
        self.zoom_canvas.bind("<ButtonPress-1>", self.start_zoom_roi)
        self.zoom_canvas.bind("<B1-Motion>", self.draw_zoom_roi_motion)
        self.zoom_canvas.bind("<ButtonRelease-1>", self.finish_zoom_roi)

        self.zoom_frame = None
        self.update_frame()

    def start_screen_roi(self, event):
        self.drawing_screen = True
        self.screen_start = (event.x, event.y)

    def draw_screen_roi_motion(self, event):
        if self.drawing_screen:
            self.screen_end = (event.x, event.y)

    def finish_screen_roi(self, event):
        if not self.drawing_screen:
            return
        self.drawing_screen = False
        x1, y1 = self.screen_start
        x2, y2 = event.x, event.y
        self.screen_roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print(f"Screen ROI: {self.screen_roi}")

    def start_zoom_roi(self, event):
        self.drawing_zoom = True
        self.zoom_start = (event.x, event.y)

    def draw_zoom_roi_motion(self, event):
        if self.drawing_zoom:
            self.zoom_end = (event.x, event.y)

    def finish_zoom_roi(self, event):
        if not self.drawing_zoom:
            return
        self.drawing_zoom = False
        x1, y1 = self.zoom_start
        x2, y2 = event.x, event.y
        self.zoomed_roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print(f"Zoomed ROI: {self.zoomed_roi}")

    def extract_temp(self, zoomed_frame):
        x1, y1, x2, y2 = self.zoomed_roi
        roi = zoomed_frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_value = self.threshold_value.get()
        _, thresh = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Debug View (Thresh)", thresh)

        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(thresh, config=config)
        clean = ''.join(c for c in text if c.isdigit() or c == '.')

        print(f"OCR Raw: '{text.strip()}' | Clean: '{clean}'")

        return clean if clean else "--"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame, (640, 480))
        self.frame = frame.copy()

        if self.screen_roi:
            x1, y1, x2, y2 = self.screen_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            screen_crop = self.frame[y1:y2, x1:x2]
            zoomed = cv2.resize(screen_crop, (640, 480), interpolation=cv2.INTER_NEAREST)
            self.zoom_frame = zoomed.copy()

            if self.zoomed_roi:
                zx1, zy1, zx2, zy2 = self.zoomed_roi
                cv2.rectangle(zoomed, (zx1, zy1), (zx2, zy2), (0, 255, 0), 2)
                reading = self.extract_temp(self.zoom_frame)
                tag = self.tag.get() or "Temp"
                self.label.config(text=f"{tag}: {reading} °C")

            img_zoom = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
            img_zoom = ImageTk.PhotoImage(Image.fromarray(img_zoom))
            self.zoom_canvas.create_image(0, 0, anchor=NW, image=img_zoom)
            self.zoom_canvas.image = img_zoom

        img_main = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_main = ImageTk.PhotoImage(Image.fromarray(img_main))
        self.canvas.create_image(0, 0, anchor=NW, image=img_main)
        self.canvas.image = img_main

        self.root.after(200, self.update_frame)


if __name__ == "__main__":
    root = Tk()
    app = TempMonitor(root)
    root.mainloop()
