import os

from PIL import Image, ImageOps

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))


def main(folder):
    for filename in os.listdir(folder):
        if not (filename.startswith("horizontalFlipped_") or filename.startswith("verticalFlipped_") or filename.startswith("90Rotated_") or filename.startswith("180Rotated_") or filename.startswith("270Rotated_")):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path)
            image_mirror_path = os.path.join(folder, "270Rotated_"+filename)
            image_mirror = image.rotate(270)
            rgb = image_mirror.convert('RGB')
            try:
                rgb.save(image_mirror_path)
            except:
                print(filename)


if __name__ == '__main__':
    main(folder=ROOT_DIR + '/dataset_kaggle/train/COVID-19/')
