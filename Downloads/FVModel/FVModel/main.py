import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# Class mappings
class_mapping = {
    0: 'buttondown_shirt',
    1: 'capri',
    2: 'dhoti_shalwar',
    3: 'kurta',
    4: 'plazzo',
    5: 'shalwar',
    6: 'short_kurti',
    7: 'straight_shirt',
    8: 'trouser'
}

# Labels to class conversion
label_to_class = {value: key for key, value in class_mapping.items()}

# Function to load the model
def load_model(model_path, num_classes):
    base_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    model_weights = torch.load(model_path, map_location=torch.device('cpu'))
    base_model.load_state_dict(model_weights)
    base_model.eval()
    return base_model

# Prepare image for model
def prepare_image(image):
    transform_pipeline = T.Compose([T.ToTensor()])
    image_tensor = transform_pipeline(image)
    return image_tensor, image

# Get the mask of the dress region
def get_dress_mask(model, image, confidence_threshold=0.3):
    image_tensor, original_image = prepare_image(image)
    image_tensor_batch = image_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor_batch)

    predicted_scores = predictions[0]['scores'].cpu().numpy()
    predicted_labels = predictions[0]['labels'].cpu().numpy()
    predicted_masks = predictions[0]['masks'].cpu().numpy()

    valid_indices = predicted_scores >= confidence_threshold
    predicted_scores = predicted_scores[valid_indices]
    predicted_labels = predicted_labels[valid_indices]
    predicted_masks = predicted_masks[valid_indices]

    if len(predicted_labels) == 0:
        return None, None

    max_confidence_index = predicted_scores.argmax()
    best_label = predicted_labels[max_confidence_index]
    best_mask = predicted_masks[max_confidence_index]

    binary_mask = (best_mask[0] > 0.5).astype(np.uint8) * 255
    return binary_mask, np.array(original_image)

# Overlay fabric pattern on the dress
def overlay_fabric_on_dress(original_image, binary_mask, fabric_pattern):
    binary_mask = binary_mask.astype(np.uint8)
    binary_mask[binary_mask > 0] = 1
    fabric_resized = cv2.resize(fabric_pattern, (original_image.shape[1], original_image.shape[0]))

    image_with_mask_removed = original_image.copy()
    image_with_mask_removed[binary_mask == 1] = [255, 255, 255]

    fabric_overlay_result = image_with_mask_removed.copy()
    for channel in range(3):
        fabric_overlay_result[:, :, channel] = np.where(
            binary_mask == 1, fabric_resized[:, :, channel], image_with_mask_removed[:, :, channel]
        )
    return fabric_overlay_result

# Blend the images
def blend_images(original_image, overlaid_image, mask, alpha=0.7):
    mask = mask.astype(np.float32)
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    mask_expanded = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

    blended_result = original_image * (1 - mask_expanded * alpha) + overlaid_image * (mask_expanded * alpha)
    return blended_result.astype(np.uint8)

# Main function to process images
def process_images(model_path, content_image, design_patch, confidence_threshold=0.3, alpha=0.7):
    num_classes = len(class_mapping) + 1
    model = load_model(model_path, num_classes)

    content_image_np = np.array(content_image)
    design_patch_np = np.array(design_patch)

    dress_mask, original_image_np = get_dress_mask(model, content_image, confidence_threshold)
    if dress_mask is None:
        return None

    fabric_pattern = cv2.cvtColor(design_patch_np, cv2.COLOR_RGB2BGR)
    original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)

    overlay_result = overlay_fabric_on_dress(original_image_np, dress_mask, fabric_pattern)
    final_output = blend_images(original_image_np, overlay_result, dress_mask, alpha)
    return final_output
