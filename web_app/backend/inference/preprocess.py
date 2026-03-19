from torchvision import transforms
from PIL import Image

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(img, crop_size=128):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    if img.size != (crop_size, crop_size):
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
    
    transform = get_transforms()
    return transform(img)
