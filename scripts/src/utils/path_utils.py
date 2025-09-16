import os

def create_output_dirs(dir_output):
    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(os.path.join(dir_output, "train", "good"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "valid", "good", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "valid", "good", "label"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "valid", "Ungood", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "valid", "Ungood", "label"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "test", "good", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "test", "good", "label"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "test", "Ungood", "img"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "test", "Ungood", "label"), exist_ok=True)

