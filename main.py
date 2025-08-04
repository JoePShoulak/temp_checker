import cv2
import numpy as np
import os

def find_screen_roi(frame, min_area=15000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_roi = None
    max_area = 0

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect = h / w if w > 0 else 0

        if len(approx) == 4 and area >= min_area and 1.2 < aspect < 2.5:
            if area > max_area:
                best_roi = (x, y, w, h)
                max_area = area

    return best_roi

def letterbox_image(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

# -------------------- Main --------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    cv2.namedWindow("Thermometer Screen ROI", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("T1%", "Thermometer Screen ROI", 25, 99, lambda x: None)
    cv2.createTrackbar("T2%", "Thermometer Screen ROI", 50, 99, lambda x: None)
    cv2.createTrackbar("T3%", "Thermometer Screen ROI", 75, 99, lambda x: None)

    screen_roi = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if screen_roi is None:
                candidate = find_screen_roi(frame)
                if candidate:
                    screen_roi = candidate
                    print(f"Locked ROI: {screen_roi}")

            display = frame.copy()

            if screen_roi:
                x, y, w, h = screen_roi

                # Subdivision slider positions
                t1 = cv2.getTrackbarPos("T1%", "Thermometer Screen ROI")
                t2 = cv2.getTrackbarPos("T2%", "Thermometer Screen ROI")
                t3 = cv2.getTrackbarPos("T3%", "Thermometer Screen ROI")
                cuts = sorted([t1, t2, t3])
                row_positions = [0] + cuts + [100]

                # Draw screen box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw subdivisions
                for i in range(4):
                    y1 = y + int(h * row_positions[i] / 100)
                    y2 = y + int(h * row_positions[i + 1] / 100)
                    cv2.rectangle(display, (x, y1), (x + w, y2), (255, 0, 0), 1)
                    cv2.putText(display, f"T{i+1}", (x + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Resize display output without distorting the drawn boxes
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect("Thermometer Screen ROI")
            except:
                win_w, win_h = 1280, 720

            padded_display = letterbox_image(display, win_w, win_h)
            cv2.imshow("Thermometer Screen ROI", padded_display)

            key = cv2.waitKey(1)

            if key == 27:  # ESC
                break

            elif key == ord('s') and screen_roi:
                print("[+] Saving training images...")
                x, y, w, h = screen_roi
                t1 = cv2.getTrackbarPos("T1%", "Thermometer Screen ROI")
                t2 = cv2.getTrackbarPos("T2%", "Thermometer Screen ROI")
                t3 = cv2.getTrackbarPos("T3%", "Thermometer Screen ROI")
                cuts = sorted([t1, t2, t3])
                row_positions = [0] + cuts + [100]

                for i in range(4):
                    y1 = y + int(h * row_positions[i] / 100)
                    y2 = y + int(h * row_positions[i + 1] / 100)
                    cropped = frame[y1:y2, x:x+w]
                    cv2.imshow("Digit", cropped)
                    label = input(f"Label for T{i+1} (0–9 or skip): ")
                    cv2.destroyWindow("Digit")

                    if label.isdigit() and 0 <= int(label) <= 9:
                        path = f"dataset/{label}/"
                        os.makedirs(path, exist_ok=True)
                        count = len(os.listdir(path))
                        filename = f"{path}/{count:04}.png"
                        cv2.imwrite(filename, cropped)
                        print(f"Saved to {filename}")
                    else:
                        print("Skipped.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
