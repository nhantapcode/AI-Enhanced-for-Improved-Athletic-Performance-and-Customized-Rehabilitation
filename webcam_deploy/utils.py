import os
from typing import Any
import cv2
import mlflow
import mediapipe as mp
import torchvision.transforms as T


def build_transforms(is_train=True):
    list_transforms = []

    if is_train:
        list_transforms.extend([
            T.Resize((224, 224)),  # Resize ảnh về kích thước cố định
            T.RandomHorizontalFlip(),  # Lật ngang ngẫu nhiên (augmentation)
            T.RandomRotation(10),  # Xoay ảnh trong khoảng [-10, 10] độ
        ])
    else:
        list_transforms.append(
            T.Resize((224, 224))  # Resize ảnh về kích thước cố định
        )
    
    # Các bước chung cho cả train và validation
    list_transforms.extend([
        T.ToTensor(),  # Chuyển ảnh sang tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Chuẩn hóa giá trị pixel
    ])

    transform = T.Compose(list_transforms)
    return transform

def process_video(input_path, output_path):
    """
    Parameters:
        input_path (str): Đường dẫn đến video đầu vào
        output_path (str): Đường dẫn đến video đầu ra
    
    Returns:
        Trả về video sau khi xử lý được lưu tại output_path,video sẽ được lưu ở định dạng mp4
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {input_path}")
        return
    
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            
            out.write(frame)
        
        # Giải phóng tài nguyên
        cap.release()
        out.release()
        print(f"Đã xử lý xong: {input_path} -> {output_path}")

def create_experiment(name: str, artifact_location: str, tags: dict[str, Any]):
    
    try:
        exp_id = mlflow.create_experiment(
            name=name,
            artifact_location=artifact_location,
            tags=tags
        )
    
    except Exception as e:
        print(f"Experiment {name} already exists")
        exp_id = mlflow.get_experiment_by_name(name).experiment_id

    return exp_id
