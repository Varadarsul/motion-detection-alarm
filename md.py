import cv2
import numpy as np
import imutils
import winsound
# Set the motion detection threshold
motion_threshold = 200
# Initialize the first frame
first_frame = None
# Start the video capture
cap = cv2.VideoCapture(0)
# Loop over the video frames
while True:
    # Capture the next frame
    ret, frame = cap.read()
    # If the frame is not captured, break the loop
    if not ret:
        break
    # Resize the frame
    frame = imutils.resize(frame, width=640, height=480)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # If the first frame is None, initialize it
    if first_frame is None:
        first_frame = gray
    # Calculate the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    # Threshold the frame delta
    thresh = cv2.threshold(frame_delta, motion_threshold, 255, cv2.THRESH_BINARY)[1]
    # Find the contours in the thresholded image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # If there are any contours, motion has been detected
    if len(contours) > 0:
        # Trigger the alarm
        winsound.Beep(1000, 1000)
        # Draw a rectangle around the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Display the frame
    cv2.imshow("Motion Detection Alarm", frame)
    # Press the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture
cap.release()
# Close all windows
cv2.destroyAllWindows()