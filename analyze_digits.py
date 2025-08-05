import cv2
import numpy as np

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

# Merge boxes that are close (within distance)
def boxes_close(a, b, dist):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return np.hypot(dx, dy) <= dist

def merge_close_boxes(boxes, dist):
    merged = []
    while boxes:
        base = boxes.pop(0)
        i = 0
        while i < len(boxes):
            if boxes_close(base, boxes[i], dist):
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

# ---------------- Parameters ----------------
MIN_AREA = 60
MAX_AREA = 1000
MIN_MERGED_AREA = 100     # Reject merged boxes smaller than this
FINAL_MIN_AREA = 200      # Final rejection of small merged boxes
MERGE_DISTANCE = 5
# --------------------------------------------

# Load image
img = cv2.imread("screen.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Connected components detection
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Store valid boxes
boxes = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if MIN_AREA <= area <= MAX_AREA:
        boxes.append([x, y, x + w, y + h])

# Show original areas
print("\n[Original Box Areas]")
for i, (x1, y1, x2, y2) in enumerate(boxes):
    area = (x2 - x1) * (y2 - y1)
    print(f"[{i+1}] area: {area}")

# Merge overlapping and nearby boxes
merged_boxes = merge_boxes(boxes)
merged_boxes = merge_close_boxes(merged_boxes, MERGE_DISTANCE)

# Filter boxes by intermediate threshold
intermediate_boxes = []
for box in merged_boxes:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area >= MIN_MERGED_AREA:
        intermediate_boxes.append(box)

# Final filtering
filtered_boxes = []
for box in intermediate_boxes:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area >= FINAL_MIN_AREA:
        filtered_boxes.append(box)

# Show areas after final filtering
print("\n[Final Filtered Box Areas]")
output = img.copy()
for i, (x1, y1, x2, y2) in enumerate(filtered_boxes):
    area = (x2 - x1) * (y2 - y1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
    print(f"[{i+1}] area: {area}")

# Display results
cv2.imshow("Filtered Merged Digit Boxes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
