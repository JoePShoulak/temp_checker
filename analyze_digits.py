import cv2
import numpy as np

# ---------------- Parameters ----------------
MIN_AREA = 60
MAX_AREA = 1000
FINAL_MIN_AREA = 200
MERGE_DISTANCE = 5
WHITE_THRESHOLD = 200
ROW_Y_TOLERANCE = 10  # pixels to consider same row
# --------------------------------------------

# Segment bitstring -> digit
SEGMENT_DIGITS = {
    "1111110": "0",
    "0110000": "1",
    "1101101": "2",
    "1111001": "3",
    "0110011": "4",
    "1011011": "5",
    "1011111": "6",
    "1110000": "7",
    "1111111": "8",
    "1111011": "9",
}

def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

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

def overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def is_close(a, b, dist=MERGE_DISTANCE):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return np.hypot(dx, dy) <= dist

def has_white_pixels_along_line(gray, start, direction, bounds):
    x, y = start
    x1, y1, x2, y2 = bounds
    if direction == "up":
        for yy in range(y, y1 - 1, -1):
            if gray[yy, x] > WHITE_THRESHOLD:
                return True
    elif direction == "down":
        for yy in range(y, y2):
            if gray[yy, x] > WHITE_THRESHOLD:
                return True
    elif direction == "left":
        for xx in range(x, x1 - 1, -1):
            if gray[y, xx] > WHITE_THRESHOLD:
                return True
    elif direction == "right":
        for xx in range(x, x2):
            if gray[y, xx] > WHITE_THRESHOLD:
                return True
    return False

def has_white_between_dots(gray, top, bottom):
    cx = top[0]
    y1 = top[1]
    y2 = bottom[1]
    for y in range(y1, y2 + 1):
        if gray[y, cx] > WHITE_THRESHOLD:
            return True
    return False

# ---------------- Main ----------------

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

# Sort all boxes top-to-bottom, then left-to-right
filtered_boxes.sort(key=lambda b: (b[1], b[0]))

# Remove the first box globally
if filtered_boxes:
    filtered_boxes = filtered_boxes[1:]

# Group remaining boxes into rows
rows = []
for box in filtered_boxes:
    x1, y1, x2, y2 = box
    placed = False
    for row in rows:
        if abs(y1 - row[0][1]) <= ROW_Y_TOLERANCE:
            row.append(box)
            placed = True
            break
    if not placed:
        rows.append([box])

# Draw boxes on image
output = img.copy()
for box in filtered_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Process and print results row-by-row
for row in rows:
    row.sort(key=lambda b: b[0])  # sort left to right
    number_str = ""
    for box in row:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        top_y = y1 + (y2 - y1) // 3
        bot_y = y1 + 2 * (y2 - y1) // 3
        top_dot = (cx, top_y)
        bot_dot = (cx, bot_y)

        # Segment detections via cardinal lines
        top_up = has_white_pixels_along_line(gray, top_dot, "up", (x1, y1, x2, y2))
        top_left = has_white_pixels_along_line(gray, top_dot, "left", (x1, y1, x2, y2))
        top_right = has_white_pixels_along_line(gray, top_dot, "right", (x1, y1, x2, y2))

        bot_down = has_white_pixels_along_line(gray, bot_dot, "down", (x1, y1, x2, y2))
        bot_left = has_white_pixels_along_line(gray, bot_dot, "left", (x1, y1, x2, y2))
        bot_right = has_white_pixels_along_line(gray, bot_dot, "right", (x1, y1, x2, y2))

        center = has_white_between_dots(gray, top_dot, bot_dot)

        bitstring = (
            ("1" if top_up else "0") +
            ("1" if top_right else "0") +
            ("1" if bot_right else "0") +
            ("1" if bot_down else "0") +
            ("1" if bot_left else "0") +
            ("1" if top_left else "0") +
            ("1" if center else "0")
        )
        digit = SEGMENT_DIGITS.get(bitstring, "?")
        number_str += digit

    # Convert to number and divide by 10
    if "?" not in number_str and number_str:
        result = int(number_str) / 10
        print(result)
    else:
        print("?", number_str)

# Show image with bounding boxes
cv2.imshow("Final Digit Bounding Boxes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
