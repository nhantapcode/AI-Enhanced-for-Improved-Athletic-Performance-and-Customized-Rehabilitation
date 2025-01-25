import os
from utils import process_video

def main() -> None:
    # Đường dẫn thư mục chính
    base_dir = "data"

    # Danh sách các bài tập
    exercises = ["barbell_biceps_curl", "hammer_curl"]
        # Duyệt qua từng bài tập và xử lý
    for exercise in exercises:
        raw_dir = os.path.join(base_dir, exercise, "raw")
        processed_dir = os.path.join(base_dir, exercise, "processed_video")

        # Tạo thư mục processed_video nếu chưa tồn tại
        os.makedirs(processed_dir, exist_ok=True)

        for file_name in os.listdir(raw_dir):
            if file_name.endswith(".mp4") or file_name.endswith(".mov"):  # Chỉ xử lý file .mp4
                input_path = os.path.join(raw_dir, file_name)
                output_path = os.path.join(processed_dir, file_name)
                process_video(input_path, output_path)


if __name__ == "__main__":
    main()