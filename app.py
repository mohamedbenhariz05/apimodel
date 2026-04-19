import warnings
warnings.filterwarnings("ignore")

import os
import io
import base64
import threading
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

# Mapping for AI Agent prompts to provide better context
CLASSES_AR = [
    "زيتون — أكاروس الزيتون (Aculus olearius)",
    "زيتون — سليم وممتاز (Healthy)",
    "زيتون — مرض عين الطاووس (Peacock spot)",
    "نخيل — اللفحة السوداء (Black scorch)",
    "نخيل — مرض البيّوض (Fusarium wilt)",
    "نخيل — سليم وممتاز (Healthy)",
    "نخيل — تبقع الأوراق (Leaf spots)",
    "نخيل — نقص المغنيسيوم (Magnesium def)",
    "نخيل — نقص المنجنيز (Manganese def)",
    "نخيل — الحشرة القشرية البيضاء (Parlatoria)",
    "نخيل — نقص البوتاسيوم (Potassium def)",
    "نخيل — لفحة الجريد (Rachis blight)"
]

CONFIDENCE_THRESHOLD = 0.60
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "student_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# MODEL
# =======================

def load_model() -> torch.jit.ScriptModule:
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    # Compile to TorchScript for faster, thread-safe inference
    scripted = torch.jit.script(model)
    return scripted

model = load_model()
_model_lock = threading.Lock()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =======================
# AI AGENT
# =======================

def get_ai_advice(class_idx: int) -> str:
    """Gets agricultural advice from the AI agent in Tunisian Arabic."""
    disease_desc = CLASSES_AR[class_idx]
    
    # If healthy, no need for disease details
    if "سليم" in disease_desc or "Healthy" in disease_desc:
        return f"النموذج يقول أن النبتة: {disease_desc}. تبدو بحالة ممتازة! ما تستحقش دواء."
    
    try:
        from g4f.client import Client
        client = Client()
        prompt = (
            f"You are an expert agricultural AI in Tunisia. The user has uploaded an image of a plant leaf "
            f"and our computer vision model diagnosed it as: '{disease_desc}'. "
            f"Please provide detailed information about this disease, including its causes, symptoms, "
            f"and recommend treatment or prevention methods. Be clear, practical, and concise. "
            f"IMPORTANT: You must provide your complete response in Tunisian Arabic dialect (الدارجة التونسية)."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while contacting the AI Agent: {str(e)}"

# =======================
# PREDICTION
# =======================

def predict(image: Image.Image) -> dict:
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with _model_lock:
        with torch.no_grad():
            output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    top_prob = float(probs.max())
    top_idx = int(probs.argmax())

    if top_prob < CONFIDENCE_THRESHOLD:
        return {"valid": False, "message": "Not a leaf or confidence too low"}

    # Get AI advice
    ai_advice = get_ai_advice(top_idx)

    top3 = sorted(
        [{"class": CLASSES[i], "prob": round(float(probs[i]), 4)} for i in range(len(CLASSES))],
        key=lambda x: x["prob"], reverse=True
    )[:3]

    return {
        "valid": True,
        "disease": CLASSES[top_idx],
        "confidence": round(top_prob * 100, 2),
        "top3": top3,
        "ai_advice": ai_advice
    }

# =======================
# FLASK APP
# =======================

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field (base64-encoded)"}), 400

    raw = data["image"]
    if not isinstance(raw, str):
        return jsonify({"error": "'image' must be a base64 string"}), 400

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception:
        return jsonify({"error": "Invalid base64 encoding"}), 400

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image exceeds 10 MB limit"}), 413

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot decode image: {e}"}), 422

    try:
        result = predict(image)
    except Exception as e:
        app.logger.exception("Inference failed")
        return jsonify({"error": "Inference error", "detail": str(e)}), 500

    return jsonify(result)

# Dev entrypoint only — use Gunicorn in production
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
