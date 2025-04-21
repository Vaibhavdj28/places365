import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Load the model
model_file = r'C:\Github Repo\places365\resnet18_places365.pth.tar'
if not os.access(model_file, os.W_OK):
    print('Model file not found!')

model = models.resnet18(num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load class labels
classes = list()
with open('C:\Github Repo\places365\categories_places365.txt') as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])

# Function to predict
def predict_room(image_path):
    img = Image.open(image_path).convert('RGB')
    input_img = transform(img).unsqueeze(0)

    # Inference
    output = model(input_img)
    _, preds = torch.max(output, 1)
    prediction = classes[preds.item()]
    return prediction

# Test
if __name__ == '__main__':
    image_path = r'C:\Users\vaibh\Downloads\istockphoto-1347784849-612x612.jpg'  # Change this!
    prediction = predict_room(image_path)
    print(f'Predicted room scene: {prediction}')
