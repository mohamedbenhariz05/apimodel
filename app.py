import warnings
warnings.filterwarnings("ignore")

import os
import io
import base64
import json
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify

# =======================
# CONFIG
# =======================

CLASSES = [
    "olive_aculus_olearius", "olive_healthy", "olive_peacock_spot",
    "palm_black_scorch", "palm_fusarium_wilt", "palm_healthy",
    "palm_leaf_spots", "palm_magnesium_deficiency", "palm_manganese_deficiency",
    "palm_parlatoria_blanchardi", "palm_potassium_deficiency", "palm_rachis_blight"
]

CONFIDENCE_THRESHOLD = 0.60
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_best.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# MODEL (loaded once at startup)
# =======================

def load_model():
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =======================
# PREDICTION
# =======================

def predict(image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    top_prob = float(probs.max())
    top_class = int(probs.argmax())

    if top_prob < CONFIDENCE_THRESHOLD:
        return {"valid": False, "message": "Not a leaf"}

    top3 = sorted(
        [{"class": CLASSES[i], "prob": float(probs[i])} for i in range(len(CLASSES))],
        key=lambda x: x["prob"], reverse=True
    )[:3]

    return {
        "valid": True,
        "disease": CLASSES[top_class],
        "confidence": round(top_prob * 100, 2),
        "top3": top3
    }

# =======================
# FLASK API
# =======================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field (base64)"}), 400

    try:
        image_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    result = predict(image)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)