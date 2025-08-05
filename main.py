import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam. Is it in use?")
    exit()

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    draw_reticle(frame)
    preprocessed = preprocess(frame)

    THRESH_MIN = 25
    THRESH_MAX = 150
    edges = cv2.Canny(preprocessed, THRESH_MIN, THRESH_MAX)

    cv2.imshow("Original", frame)
    cv2.imshow("Preprocessed", preprocessed)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
