from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw, ImageFont
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
    confidence_threshold=0.55,
    device=device.type
)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_and_classify(image_path):
    """
    Detect acne regions and classify their types with coordinate mapping to the original image.
    """

    # Load the original image
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size

    # Resize the image to 224x224 for detection
    resized_image = original_image.resize((448, 448))
    resized_width, resized_height = resized_image.size

    # Perform object detection on the resized image
    result = get_sliced_prediction(
        resized_image,
        yolo_model,
        slice_height=91*2,
        slice_width=91*2,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
    )

    # Get detected predictions
    predictions = result.object_prediction_list
    print(f"Total predictions: {len(predictions)}")

    draw = ImageDraw.Draw(original_image)  # Draw on the original image
    bounding_boxes = []

    for prediction in predictions:
        # Get bounding box coordinates from resized image
        x1, y1, x2, y2 = map(lambda coord: round(coord, 2), prediction.bbox.to_xyxy())

        # Mapping coordinates from resized (224x224) image back to original image
        x1 = int((x1 / resized_width) * original_width)
        y1 = int((y1 / resized_height) * original_height)
        x2 = int((x2 / resized_width) * original_width)
        y2 = int((y2 / resized_height) * original_height)

        # Crop detected region from the original image
        crop = original_image.crop((x1, y1, x2, y2))

        # Resize the cropped region to 224x224 for classification
        img_transformed = transform(crop.resize((224, 224))).unsqueeze(0).to(device)

        # Perform classification on the cropped region
        with torch.no_grad():
            output = classification_model(img_transformed)
            probabilities = torch.softmax(output, dim=1)
            top_prob, top_class = probabilities.topk(1, dim=1)
            top_prob = top_prob.item()
            top_class = top_class.item()
            class_name_en = class_labels[top_class]

        # Calculate confidence percentage
        percentage_conf = f"{top_prob * 100:.0f}"

        # Create label for the bounding box
        label = f"{class_name_en} ({percentage_conf}%)"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Draw bounding box and label on the original image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), label, fill="white", font=font)

        # Append bounding box data with class label and confidence
        bounding_boxes.append({
            "class_id": class_name_en,
            "cords": [x1, y1, x2, y2],
            "percentage_conf": percentage_conf
        })

    return bounding_boxes, original_image, len(predictions)
