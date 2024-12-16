import cv2
import csv
import mediapipe as mp
import os


def extract_landmarks(video_path, output_csv):
    """
    Extracts 33 pose landmarks (x, y, z) for each frame of a video and saves them to a CSV file.

    Parameters:
        video_path (str): Path to the input video file.
        output_csv (str): Path to the output CSV file.

    Returns:
        None
    """

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    landmarks_list = []

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks on the frame (optional)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Save landmarks for the current frame
            if results.pose_landmarks:
                frame_landmarks = [frame_count]  # Save frame index
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.extend(
                        [landmark.x, landmark.y, landmark.z]
                    )  # Save x, y, z
                landmarks_list.append(frame_landmarks)

            # Display the video with landmarks
            cv2.imshow("Video", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)

        # Write header: Frame, x_0, y_0, z_0, ..., x_32, y_32, z_32
        header = ["Frame"]
        for i in range(33):  # 33 landmarks
            header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

        writer.writerows(landmarks_list)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Landmarks saved to {output_csv}")


def main():
    folder_path = "../data/preprocess_data/"
    video_path = "tap_squat_dung.mp4"
    path = os.path.normpath(os.path.join(folder_path, video_path))

    video_name = video_path.rsplit(".", 1)[0]

    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    output_csv = os.path.join(output_folder, f"{video_name}.csv")

    extract_landmarks(path, output_csv)


if __name__ == "__main__":
    main()
