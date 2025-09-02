import os
from PIL import Image
from torchvision import transforms

def load_images_from_folder(folder, size=(256, 256)):
    images = []
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            images.append(img)
    return images
