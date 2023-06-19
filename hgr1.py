import cv2
import numpy as np

def detect_hand(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in the edge image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box of the largest contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the hand region from the image
    hand = image[y:y + h, x:x + w]

    # Find the center of the hand region
    (cx, cy) = (x + w // 2, y + h // 2)

    # Calculate the angle of the hand
    angle = cv2.angle(largest_contour)

    # If the angle is greater than 180 degrees, subtract 360 degrees
    if angle > 180:
        angle -= 360

    # Classify the hand as right or left
    if angle > 0 and angle < 90:
        return "right"
    elif angle > 90 and angle < 180:
        return "left"
    else:
        return "unknown"

