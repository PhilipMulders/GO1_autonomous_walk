import cv2
import numpy as np

# Function to estimate the distance to a contour using pinhole camera model
def estimate_distance(contour, known_height, focal_length, frame_height):
    """
    Estimate the distance to a contour using the known height, width of the object and the focal length.
    
    :param contour: The detected contour.
    :param known_height: The real-world height of the object (in meters).
    :param focal_length: The focal length of the camera (in pixels).
    :param frame_height: The height of the frame (in pixels).
    :return: Estimated distance (in meters).
    """
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the distance using the pinhole camera model:
    size = (h + w) / 2
    distance = (known_height * focal_length) / size
    
    return distance

# Function to process the frame and detect objects
def detect_objects(frame, focal_length, known_height):
    """
    Detect contours in the frame and estimate distances for obstacles.
    
    :param frame: The input video frame.
    :param focal_length: Focal length of the camera in pixels.
    :param known_height: Known real-world height of target objects in meters.
    :return: Processed frame with detected objects and distance estimations.
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # canny edge detection. 
    # Compute median of pixel intensities and create an upper and lower thresh
    median_intensity = np.median(gray)
    lower_thresh = int(max(0, 0.66 * median_intensity))
    upper_thresh = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for distance tracking
    closest_contour = None
    min_distance = float('inf')

    # Iterate through each detected contour
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Filter out small contours to reduce noise
            # Estimate distance to the contour
            distance = estimate_distance(contour, known_height, focal_length, frame.shape[0])

            # Find the closest contour
            if distance < min_distance:
                min_distance = distance
                closest_contour = contour

            # Visualize the contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # After all contours are processed, if the closest one was found, annotate it with distance
    if closest_contour is not None:
        x, y, w, h = cv2.boundingRect(closest_contour)
        cv2.putText(frame, f"Distance: {min_distance:.1f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

# Main function to capture video and process frames
def main():
    cap = cv2.VideoCapture(1) # I use DroidCam, so (1) is here. Webcam typically is at 0
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Known values for distance estimation (in meters)
    known_height = 0.3  # meters (adjust based on the object you're detecting)
    focal_length = 900.0  # Approximation, adjust based on testing

    frame_counter = 0  # Initialize frame counter

    print("Connected to camera! Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        frame_counter += 1  # Increment frame counter

        # Process every 4th frame for object detection and distance estimation
        if frame_counter % 4 == 0:
            processed_frame = detect_objects(frame, focal_length, known_height)

            # Show the processed frame
            cv2.imshow('Object Detection', processed_frame)

        # Break the loop when the user presses 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting.")
            break

    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
