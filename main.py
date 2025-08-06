import cv2
import numpy as np
import analyze_digits

def draw_reticle(frame):
    RETICLE_COLOR = (0, 0, 255)  # red
    RETICLE_SIZE = 5
    RETICLE_THICKNESS = -1  # filled
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), radius=RETICLE_SIZE, color=RETICLE_COLOR, thickness=RETICLE_THICKNESS)

def preprocess(frame):
    BLUR_SIZE = 5
    SIGMA_X = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale
    return cv2.GaussianBlur(frame, (BLUR_SIZE, BLUR_SIZE), SIGMA_X)  # blur to reduce noise

def make_contours(frame):
    THRESH_MIN = 25
    THRESH_MAX = 150
    return cv2.Canny(frame, THRESH_MIN, THRESH_MAX)

def find_screen_roi(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    center_point = (width // 2, height // 2)

    for cnt in contours:
        if cv2.pointPolygonTest(cnt, center_point, measureDist=False) >= 0:
            x, y, w, h = cv2.boundingRect(cnt)
            return (x, y, w, h)

def draw_screen_roi(frame, roi):
    ROI_COLOR = (0, 255, 0)  # green
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), ROI_COLOR, 2)
    return frame[y:y + h, x:x + w]

def preprocess_screen_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (frame.shape[1]*2, frame.shape[0]*2))

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )

    # Reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


######## MAIN ########
def main():
    screen_roi = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam. Is it in use?")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret: break

        preprocessed = preprocess(frame)
        edges = make_contours(preprocessed)
        cropped = None
        screen = None

        if not screen_roi: screen_roi = find_screen_roi(edges)
        if screen_roi: cropped = draw_screen_roi(frame, screen_roi)

        if cropped is not None:
            screen = preprocess_screen_roi(cropped)

        if screen is not None:
            values, digits_img = analyze_digits.analyze_digits(screen)
            for v in values:
                if v is not None:
                    print(v)
            cv2.imshow("Digits Detected", digits_img)

        draw_reticle(frame)
        cv2.imshow("Original", frame)
        cv2.imshow("Preprocessed", preprocessed)
        cv2.imshow("Edges", edges)
        if cropped is not None:
            cv2.imshow("Screen ROI", cropped)
        if screen is not None:
            cv2.imshow("Screen ROI Preprocessed", screen)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
