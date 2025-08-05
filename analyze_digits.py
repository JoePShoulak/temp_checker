import cv2
import numpy as np

# Load image
img = cv2.imread("screen.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to isolate white digits
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Connected components detection
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Store valid boxes
MIN_AREA = 60
MAX_AREA = 1000
boxes = []
for i in range(1, num_labels):  # skip background
    x, y, w, h, area = stats[i]
    if MIN_AREA <= area <= MAX_AREA:
        boxes.append([x, y, x + w, y + h])  # (x1, y1, x2, y2)

# Merge overlapping boxes
def do_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def merge_boxes(boxes):
    merged = []
    while boxes:
        base = boxes.pop(0)
        i = 0
        while i < len(boxes):
            if do_overlap(base, boxes[i]):
                # Merge and restart
                x1 = min(base[0], boxes[i][0])
                y1 = min(base[1], boxes[i][1])
                x2 = max(base[2], boxes[i][2])
                y2 = max(base[3], boxes[i][3])
                base = [x1, y1, x2, y2]
                boxes.pop(i)
                i = 0
            else:
                i += 1
        merged.append(base)
    return merged

merged_boxes = merge_boxes(boxes)

# Draw result
output = img.copy()
for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
    print(f"[{i+1}] x:{x1}, y:{y1}, w:{x2 - x1}, h:{y2 - y1}")

# Show in a live OpenCV window
cv2.imshow("Merged Digit Boxes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
