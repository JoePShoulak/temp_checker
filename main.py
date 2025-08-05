import cv2

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check that it works
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
