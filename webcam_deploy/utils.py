import os
import cv2
import mediapipe as mp

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