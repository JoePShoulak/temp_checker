import cv2
import numpy as np

# ---------------- Parameters ----------------
MIN_AREA = 60
MAX_AREA = 1000
MIN_MERGED_AREA = 100
FINAL_MIN_AREA = 200
MERGE_DISTANCE = 5
# --------------------------------------------

# Helper to compute area of a box
def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

# Merge boxes based on a condition
def merge_boxes_by_condition(boxes, condition):
    merged = []
    while boxes:
        base = boxes.pop(0)
        i = 0
        while i < len(boxes):
            if condition(base, boxes[i]):
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

# Overlap condition
def overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

# Closeness condition
def is_close(a, b, dist=MERGE_DISTANCE):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return np.hypot(dx, dy) <= dist

# Load and process image
img = cv2.imread("screen.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Initial bounding boxes
boxes = [
    [x, y, x + w, y + h]
    for x, y, w, h, area in stats[1:]
    if MIN_AREA <= area <= MAX_AREA
]

# Merge overlapping and close boxes
boxes = merge_boxes_by_condition(boxes, overlaps)
boxes = merge_boxes_by_condition(boxes, lambda a, b: is_close(a, b, MERGE_DISTANCE))

# Final filtering
filtered_boxes = [b for b in boxes if box_area(b) >= FINAL_MIN_AREA]

output = img.copy()
for i, box in enumerate(filtered_boxes):
    x1, y1, x2, y2 = box
    area = box_area(box)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Show result
cv2.imshow("Filtered Merged Digit Boxes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
