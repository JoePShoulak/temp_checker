import cv2

def draw_reticle(frame):
    RETICLE_COLOR = (0, 0, 255) # red
    RETICLE_SIZE = 5
    RETICLE_THICKNESS = -1 # filled
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), radius=RETICLE_SIZE, color=RETICLE_COLOR, thickness=-RETICLE_THICKNESS)

def preprocess(frame):
    BLUR_SIZE = 5
    SIGMA_X = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale
    return cv2.GaussianBlur(frame, (BLUR_SIZE, BLUR_SIZE), SIGMA_X) # blur to reduce noise

def make_contours(frame):
    THRESH_MIN = 25
    THRESH_MAX = 150
    return cv2.Canny(frame, THRESH_MIN, THRESH_MAX)

def find_roi(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    center_point = (width // 2, height // 2)

    for cnt in contours:
        if cv2.pointPolygonTest(cnt, center_point, measureDist=False) >= 0:
            x, y, w, h = cv2.boundingRect(cnt)
            return (x, y, w, h)
        
def draw_roi(frame, roi):
        ROI_COLOR = (0, 255, 0) # green
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), ROI_COLOR, 2)
        return frame[y:y + h, x:x + w]

######## MAIN ########
def main():
    roi = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam. Is it in use?")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret: break

        draw_reticle(frame)
        preprocessed = preprocess(frame)
        edges = make_contours(preprocessed)
        cropped = None

        if not roi: roi = find_roi(edges)
        if roi: cropped = draw_roi(frame, roi)

        cv2.imshow("Original", frame)
        cv2.imshow("Preprocessed", preprocessed)
        cv2.imshow("Edges", edges)
        if cropped is not None: cv2.imshow("Screen ROI", cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
