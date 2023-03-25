import cv2

# Open the video file
video = cv2.VideoCapture("data/test_0.mp4")

# Get the frame rate and size of the input video
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter("output_video.mp4", fourcc, fps, frame_size, isColor=False)

# Loop through the frames of the input video
while True:
    # Read a frame from the video
    ret, frame = video.read()
    if not ret:
        break

    # Apply Gaussian blur to the input frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Write the output frame to the output video
    output_video.write(edges)

    # Display the output frame
    cv2.imshow("Canny Edge Detection", edges)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
video.release()
output_video.release()
cv2.destroyAllWindows()
