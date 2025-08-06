import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DigitParams:
    """Parameters controlling digit extraction."""
    min_area: int = 60
    max_area: int = 1000
    final_min_area: int = 200
    merge_distance: int = 5
    white_threshold: int = 200
    row_y_tolerance: int = 10  # pixels to consider same row

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

def is_close(a, b, dist):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return np.hypot(dx, dy) <= dist

def has_white_pixels_along_line(gray, start, direction, bounds, threshold):
    x, y = start
    x1, y1, x2, y2 = bounds
    if direction == "up":
        for yy in range(y, y1 - 1, -1):
            if gray[yy, x] > threshold:
                return True
    elif direction == "down":
        for yy in range(y, y2):
            if gray[yy, x] > threshold:
                return True
    elif direction == "left":
        for xx in range(x, x1 - 1, -1):
            if gray[y, xx] > threshold:
                return True
    elif direction == "right":
        for xx in range(x, x2):
            if gray[y, xx] > threshold:
                return True
    return False

def has_white_between_dots(gray, top, bottom, threshold):
    cx = top[0]
    y1 = top[1]
    y2 = bottom[1]
    for y in range(y1, y2 + 1):
        if gray[y, cx] > threshold:
            return True
    return False


def analyze_digits(gray, params: DigitParams = DigitParams()):
    """Detect seven-segment digits in a grayscale or binary image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale or binary image containing the digits.
    params : DigitParams, optional
        Tuning parameters for the detection algorithm.

    Returns
    -------
    tuple[list[float], np.ndarray]
        A tuple of detected numeric values (each divided by 10 as in the
        original script) and an output image with bounding boxes drawn.
    """

    # Ensure binary image for component analysis
    _, binary = cv2.threshold(gray, params.white_threshold, 255, cv2.THRESH_BINARY)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Initial bounding boxes
    boxes = [
        [x, y, x + w, y + h]
        for x, y, w, h, area in stats[1:]
        if params.min_area <= area <= params.max_area
    ]

    # Merge overlapping and close boxes
    boxes = merge_boxes_by_condition(boxes, overlaps)
    boxes = merge_boxes_by_condition(boxes, lambda a, b: is_close(a, b, params.merge_distance))

    # Final filtering
    filtered_boxes = [b for b in boxes if box_area(b) >= params.final_min_area]

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
            if abs(y1 - row[0][1]) <= params.row_y_tolerance:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    # Prepare output image
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)

    results = []
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
            top_up = has_white_pixels_along_line(
                gray, top_dot, "up", (x1, y1, x2, y2), params.white_threshold
            )
            top_left = has_white_pixels_along_line(
                gray, top_dot, "left", (x1, y1, x2, y2), params.white_threshold
            )
            top_right = has_white_pixels_along_line(
                gray, top_dot, "right", (x1, y1, x2, y2), params.white_threshold
            )

            bot_down = has_white_pixels_along_line(
                gray, bot_dot, "down", (x1, y1, x2, y2), params.white_threshold
            )
            bot_left = has_white_pixels_along_line(
                gray, bot_dot, "left", (x1, y1, x2, y2), params.white_threshold
            )
            bot_right = has_white_pixels_along_line(
                gray, bot_dot, "right", (x1, y1, x2, y2), params.white_threshold
            )

            center = has_white_between_dots(
                gray, top_dot, bot_dot, params.white_threshold
            )

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
            results.append(int(number_str) / 10)
        else:
            results.append(None)

    return results, output


if __name__ == "__main__":
    img = cv2.imread("screen.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = DigitParams()
    values, output = analyze_digits(gray, params)
    for v in values:
        print(v if v is not None else "?")
    cv2.imshow("Final Digit Bounding Boxes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
