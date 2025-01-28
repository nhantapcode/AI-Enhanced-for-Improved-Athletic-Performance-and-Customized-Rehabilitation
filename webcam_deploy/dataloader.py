import os
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KaggleDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        """
        Args:
            image_dirs (dict): Dictionary chứa tên class và đường dẫn đến thư mục ảnh của class đó.
                               Ví dụ: {"class_0": "path/to/class_0_images", "class_1": "path/to/class_1_images"}.
            transform (callable, optional): Transform cần áp dụng lên ảnh.
        """
        self.transform = transform
        self.samples = []  # Lưu danh sách (đường dẫn ảnh, label)

        # Truy cập thẳng vào thư mục của từng class và lấy ảnh
        for label, image_dir in enumerate(image_dirs.values()):
            for img_name in os.listdir(image_dir):
                img_path = os.path.join(image_dir, img_name)
                if os.path.isfile(img_path):  # Chỉ thêm file, không thêm folder
                    self.samples.append((img_path, label))  # Gắn nhãn tương ứng với class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Load ảnh
        if self.transform:
            image = self.transform(image)  # Áp dụng transform
        return image, label