import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam. Is it in use?")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    BLUR_SIZE = 5
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0) # blur to reduce noise

    # Show intermediate steps
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
