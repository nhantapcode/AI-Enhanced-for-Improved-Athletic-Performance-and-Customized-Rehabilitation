import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from time import time


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
)
mp_drawing = mp.solutions.drawing_utils


def detectPose(image, pose, display=True):
    # create a copy of the image
    outout_image = image.copy()
    # convert the image to RGB
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # perform pose detection
    results = pose.process(imageRGB)
    # retrieve the height and width of the input image
    height, width, _ = image.shape
    # initialize a list to store the landmarks
    landmarks = []
    # check if any landmarks are detected
    if results.pose_landmarks:
        # draw the landmarks on the output image
        mp_drawing.draw_landmarks(
            outout_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        # iterate over all the detected landmarks
        for landmark in results.pose_landmarks.landmark:
            # append the landmark coordinates to the list
            landmarks.append(
                (
                    int(landmark.x * width),
                    int(landmark.y * height),
                    (landmark.z * width),
                )
            )

    # check if the original input image and the result image are specified to be displayed
    if display:
        # specify a size of the figure
        plt.figure(figsize=[22, 22])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Input image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(outout_image[:, :, ::-1])
        plt.title("Output image")
        plt.axis("off")

        # also display the pose landmarks in 3D
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS
        )

    else:
        # return the output image and landmarks
        return outout_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # calculate the angle between the 3 landmarks
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # check if the angle is negative
    if angle < 0:
        # add 360 to the angle
        angle += 360

    # return the angle
    return angle


def classifyPose(landmarks, output_image, display=False):
    # initialize the pose label from start
    label = "Unknown"
    # specify the color (red) with which the pose will be annotated
    color = (0, 0, 255)
    # calculate the required angles for the poses
    # -------------------------------------------------------------------------------------------------------------------
    # left elbow angle between left shoulder, left elbow, and left wrist landmarks
    left_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
    )
    # right elbow angle between right shoulder, right elbow, and right wrist landmarks
    right_elbow_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
    )
    # left elbow, shoulder, and hip angle
    left_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    )
    # right elbow, shoulder, and hip angle
    right_shoulder_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    )
    # left knee, hip, and ankle angle
    left_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
    )
    # right knee, hip, and ankle angle
    right_knee_angle = calculateAngle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    )
    # -------------------------------------------------------------------------------------------------------------------
    # check if the pose is Warrior II or T pose
    # because both of them have both arms should be straight, shoulder should be at the specific angle.
    # check if both arms are straight
    if (
        left_elbow_angle > 165
        and left_elbow_angle < 195
        and right_elbow_angle > 165
        and right_elbow_angle < 195
    ):
        # check if the shoulder are at the required angle
        if (
            left_shoulder_angle > 80
            and left_shoulder_angle < 110
            and right_shoulder_angle > 80
            and right_shoulder_angle < 110
        ):
            # check if it is Warrior II pose
            # check one leg is straight
            if (
                left_knee_angle > 165
                and left_knee_angle < 195
                or right_knee_angle > 165
                and right_knee_angle < 195
            ):
                # check if the other leg is bended at the required angle
                if (
                    left_knee_angle > 90
                    and left_knee_angle < 120
                    or right_knee_angle > 90
                    and right_knee_angle < 120
                ):
                    # update the pose label
                    label = "Warrior II pose"

            # check if it is T pose
            # check if both legs are straight
            if (
                left_knee_angle > 160
                and left_knee_angle < 195
                and right_knee_angle > 160
                and right_knee_angle < 195
            ):
                # update the pose label
                label = "T pose"
    # -------------------------------------------------------------------------------------------------------------------

    # check if the pose is Tree pose
    if (
        left_knee_angle > 165
        and left_knee_angle < 195
        or right_knee_angle > 165
        and right_knee_angle < 195
    ):
        if (
            left_knee_angle > 315
            and left_knee_angle < 335
            or right_knee_angle > 25
            and right_knee_angle < 45
        ):
            label = "Tree pose"
    # -------------------------------------------------------------------------------------------------------------------
    # check if the pose is classiified successfully
    if label != "Unknown":
        # update the color to green
        color = (0, 255, 0)

    # write the label on the output image
    cv.putText(output_image, label, (20, 40), cv.FONT_HERSHEY_PLAIN, 2, color, 2)

    # check if the resultant image is specified to be displayed
    if display:
        # display the resultant image
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("output image")
        plt.axis("off")
    else:
        # return the output image
        return output_image, label


def main():
    # video_path = "yoga.mp4"
    # output_path = "pose_classification.mp4"
    pose_video = mp_pose.Pose(
        static_image_mode=False, min_detection_confidence=0.5, model_complexity=1
    )

    camera_video = cv.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)
    frame_width = int(camera_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera_video.get(cv.CAP_PROP_FPS))

    # Định nghĩa VideoWriter để lưu video
    # fourcc = cv.VideoWriter_fourcc(*'avc1')  # Codec
    # out = cv.VideoWriter(f'filename.mp4', cv.VideoWriter_fourcc('a', 'v', 'c', '1'), 25.0, (frame_width, frame_height))


    cv.namedWindow("Pose classification", cv.WINDOW_NORMAL)

    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue

        frame = cv.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        frame = cv.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks = detectPose(frame, pose_video, display=False)

        if landmarks:
            frame, label = classifyPose(landmarks, frame, display=False)

        cv.imshow("Pose classification", frame)

        # out.write(frame)

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    # release the videocapture object and close the window
    camera_video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
