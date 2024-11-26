import cv2
import numpy as np
import time

# Step 3: Capture Video Stream
# Use a webcam as the video source
#cap = cv2.VideoCapture(0)  # For IP camera, replace with the stream URL.
cap = cv2.VideoCapture("http://192.168.1.2:4747/video")

if not cap.isOpened():
    print("Error: Could not open IP camera stream.")
    exit()

def find_reddest_pixel(frame):
    """
    Find the reddest pixel based on the red channel intensity compared to the blue and green channels.
    """
    b, g, r = cv2.split(frame)
    redness = r - (b + g) // 2
    _, _, _, max_loc = cv2.minMaxLoc(redness)
    return max_loc

def find_brightest_pixel_with_loop(frame):
    """
    Find the brightest pixel by iterating through each pixel in a grayscale image.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_val = 0
    max_loc = (0, 0)
    for y in range(gray_frame.shape[0]):
        for x in range(gray_frame.shape[1]):
            if gray_frame[y, x] > max_val:
                max_val = gray_frame[y, x]
                max_loc = (x, y)
    return max_loc

while True:
    # Measure start time
    start_time = time.time()

    # Step 1: Capture the frame
    ret, frame = cap.read()
    capture_time = time.time() - start_time  # Time for capturing
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Step 2: Find the brightest spot using cv2.minMaxLoc()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, _, _, max_loc_built_in = cv2.minMaxLoc(gray_frame)
    cv2.circle(frame, max_loc_built_in, 15, (255, 255, 0), 2)  # Yellow for cv2 bright spot
    bright_spot_builtin_time = time.time() - start_time - capture_time

    # Step 3: Find the reddest pixel
    reddest_loc = find_reddest_pixel(frame)
    cv2.circle(frame, reddest_loc, 15, (0, 0, 255), 2)  # Red for reddest spot
    red_spot_time = time.time() - start_time - capture_time - bright_spot_builtin_time

    # Step 4: Find the brightest spot using a double for-loop
    brightest_loc_loop = find_brightest_pixel_with_loop(frame)
    cv2.circle(frame, brightest_loc_loop, 15, (0, 255, 255), 2)  # Cyan for loop bright spot
    bright_spot_loop_time = time.time() - start_time - capture_time - bright_spot_builtin_time - red_spot_time

    # Step 5: Display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    qcv2.imshow('Real-Time Detection', frame)

    # Measure total processing time
    total_time = time.time() - start_time

    # Print timing information
    print(f"Capture Time: {capture_time:.4f}s, Built-in Bright Spot Time: {bright_spot_builtin_time:.4f}s, "
          f"Red Spot Time: {red_spot_time:.4f}s, Loop Bright Spot Time: {bright_spot_loop_time:.4f}s, "
          f"Total Time: {total_time:.4f}s")

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()