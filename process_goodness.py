from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms
import torch.nn as nn

# Define class labels
class_labels = [
    'acne_scars', 'blackhead', 'cystic', 'flat_wart', 'folliculitis',
    'keloid', 'milium', 'papular', 'purulent', 'sebo-crystan-conglo',
    'syringoma', 'whitehead'
]

# Initialize classification model
class ClassificationModel(nn.Module):
    def __init__(self, num_classes=12):
        super(ClassificationModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load classification model
classification_model = ClassificationModel(num_classes=12)
classification_model.load_state_dict(torch.load('./Models/AcneClassification.pt', map_location=device))
classification_model.to(device)
classification_model.eval()

# Load YOLO detection model
yolo_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="./Models/AcneDetect.pt",
    confidence_threshold=0.5,
    device=device.type
)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_and_classify(image_path):
    """
    Detect acne regions and classify their types.
    """

    # Open and preprocess the image
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((224, 224))  # Resize image to fixed size for detection

    # Perform object detection
    result = get_sliced_prediction(
        pil_image,
        yolo_model,
        slice_height=91,
        slice_width=91,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
    )

    # Get detected predictions
    predictions = result.object_prediction_list
    print(f"Total predictions: {len(predictions)}")  # Debugging output

    draw = ImageDraw.Draw(pil_image)
    bounding_boxes = []

    for prediction in predictions:
        # Get bounding box coordinates and round to 2 decimal places
        x1, y1, x2, y2 = map(lambda coord: round(coord, 2), prediction.bbox.to_xyxy())

        # Crop detected region
        crop = pil_image.crop((x1, y1, x2, y2))
        img_transformed = transform(crop).unsqueeze(0).to(device)

        # Perform classification
        with torch.no_grad():
            output = classification_model(img_transformed)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_id = probabilities.argmax(1).item()

        # Get class label
        predicted_class_label = class_labels[predicted_class_id]

        # Draw bounding box with label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), predicted_class_label, fill="red")

        # Append bounding box data with class label
        bounding_boxes.append({
            "coordinates": [x1, y1, x2, y2],
            "class_label": predicted_class_label
        })

    return bounding_boxes, pil_image, len(predictions)
