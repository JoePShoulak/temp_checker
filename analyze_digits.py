import cv2
import numpy as np

# Load image
img = cv2.imread("screen.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to isolate white digits
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Connected components detection
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Draw boxes on a copy of the original image
output = img.copy()
min_area = 10
blob_count = 0

for i in range(1, num_labels):  # skip background
    x, y, w, h, area = stats[i]
    if area >= min_area:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        blob_count += 1
        print(f"[{blob_count}] x:{x}, y:{y}, w:{w}, h:{h}, area:{area}")

# Show in a live OpenCV window
cv2.imshow("Digit Blob Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
