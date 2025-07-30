import cv2
import numpy as np
import os
import collections
from math import atan2, degrees

def get_line_angle(x1, y1, x2, y2):
    angle = degrees(atan2(y2 - y1, x2 - x1))
    return angle % 180  # keep within [0, 180)

def detect_card_angle_fixed(image_path, output_dir="output_detected_angles"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    original_img = img.copy()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Probabilistic Hough Transform — gives endpoints
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) == 0:
        print(f"No lines detected in: {os.path.basename(image_path)}")
        return 0

    # Calculate angles for each line
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = get_line_angle(x1, y1, x2, y2)
        # Only consider angles in a valid range, e.g., ignore near-horizontal lines
        if 10 < angle < 170:  # avoid noise
            angles.append(angle if angle <= 90 else angle - 180)  # normalize to [-90, 90]

    if not angles:
        print(f"No valid angles found in: {os.path.basename(image_path)}")
        return 0

    # Get the dominant rotation angle (most common bin)
    hist = np.histogram(angles, bins=180, range=(-90, 90))
    bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
    dominant_angle = bin_centers[np.argmax(hist[0])]

    print(f"{os.path.basename(image_path)} → Detected angle: {dominant_angle:.2f}°")

    # Draw detected lines on image
    vis_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw center and angle line
    center_x, center_y = width // 2, height // 2
    length = min(width, height) // 2
    rad = np.radians(dominant_angle)
    end_x = int(center_x + length * np.cos(rad))
    end_y = int(center_y - length * np.sin(rad))
    cv2.line(vis_img, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
    cv2.putText(vis_img, f"Angle: {dominant_angle:.1f}°", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(image_path)
    out_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}_angle.jpg")
    cv2.imwrite(out_path, vis_img)

    return dominant_angle

def process_folder_fixed(input_folder="images_new", output_folder="output_detected_angles"):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(input_folder, filename)
            detect_card_angle_fixed(image_path, output_folder)

# Run
if __name__ == "__main__":
    process_folder_fixed("images_new", "output_detected_angles")
