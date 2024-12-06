import cv2
import os
import csv

def extract_and_label_frames(video_path, output_dir, label_file, resize_to=(64, 64), window_size=(800, 600), skip_frames=10):
    """
    Extract every nth frame from a video and manually label each frame as 'bench' or 'no_bench'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    labeled_frame_count = 0  # Track labeled frames
    with open(label_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'frame_name'])  # Header row

        # Set up the OpenCV window with a specified size
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", window_size[0], window_size[1])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            # Resize and save the frame
            frame_resized = cv2.resize(frame, resize_to)
            frame_name = f"frame_{labeled_frame_count}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame_resized)

            # Resize frame for display
            frame_resized_for_display = cv2.resize(frame, (window_size[0], window_size[1]))

            # Change the window title to show the current labeled frame number
            cv2.setWindowTitle("Frame", f"Frame {labeled_frame_count}")

            # Display the frame in the window
            cv2.imshow("Frame", frame_resized_for_display)
            key = cv2.waitKey(0)

            # Label the frame based on key press
            if key == ord('b'):  # 'b' for bench
                label = 'bench'
            elif key == ord('n'):  # 'n' for no bench
                label = 'no_bench'
            else:
                print("Invalid key pressed! Press 'b' for bench or 'n' for no_bench.")
                continue

            # Write the label to the CSV file
            writer.writerow([label, frame_name])

            frame_count += 1
            labeled_frame_count += 1
        else:
            # Increment even when skipping
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished labeling {labeled_frame_count} frames. Labels saved to {label_file}")

# Example usage
video_path = 'data/videos/nameofYourVideo.mp4'  # Replace with your video file path
output_dir = 'data/frames'  # Directory to save frames
label_file = 'data/labels.csv'  # CSV file to efficiently store labels

# Start the extraction and labeling process
extract_and_label_frames(video_path, output_dir, label_file)

