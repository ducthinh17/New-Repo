from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from convert_sahi import convert_sahi_results_to_bounding_boxes
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms
import torch.nn as nn

# Initialize classification model
class ClassificationModel(nn.Module):
    def __init__(self, num_classes=12):
        super(ClassificationModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification_model = ClassificationModel(num_classes=12)
classification_model.load_state_dict(torch.load('./Models/AcneClassification.pt', map_location=device))
classification_model.to(device)
classification_model.eval()

# Initialize detection model
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
    
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((224, 224))  # Ensure fixed input size

    result = get_sliced_prediction(
        pil_image,
        yolo_model,
        slice_height=91,
        slice_width=91,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
    )

    # Access predictions from slices
    predictions = result.object_prediction_list
    print(f"Total predictions: {len(predictions)}")  # Debugging total predictions

    draw = ImageDraw.Draw(pil_image)
    bounding_boxes = []

    for prediction in predictions:
        # Extract bounding box and class info
        x1, y1, x2, y2 = prediction.bbox.to_xyxy()
        class_id = prediction.category.name

        # Crop the region for classification
        crop = pil_image.crop((x1, y1, x2, y2))
        img_transformed = transform(crop).unsqueeze(0).to(device)

        # Classify the region
        with torch.no_grad():
            output = classification_model(img_transformed)
            probabilities = torch.softmax(output, dim=1)
            predicted = probabilities.argmax(1).item()

        # Draw bounding box and add to the list
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        bounding_boxes.append({
            "coordinates": [x1, y1, x2, y2],
            "class_id": predicted
        })

    return bounding_boxes, pil_image, len(predictions)
