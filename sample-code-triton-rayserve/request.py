import requests
import numpy as np
from PIL import Image

# Load ImageNet class names from synset.txt
with open("synset.txt") as f:
    labels = [line.strip() for line in f]

def preprocess_image(img_path: str) -> np.ndarray:
    """Load image, resize to 224×224, normalize, and return as (1,3,224,224)."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0        # normalize to [0,1]
    arr = arr.transpose(2, 0, 1)                           # HWC -> CHW
    return arr[np.newaxis, ...]                            # add batch dim

def softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to raw logits."""
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def predict(image_path: str):
    # 1. Preprocess
    batch = preprocess_image(image_path)
    
    # 2. Send request to Ray Serve endpoint
    url = "http://localhost:8080/predict_squeezenet"
    payload = {"data": batch.tolist()}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    
    # 3. Parse logits and compute probabilities
    logits = np.array(resp.json(), dtype=np.float32)
    probs = softmax(logits)
    
    # 4. Get top-5 predictions
    top5_idx = probs.argsort()[-5:][::-1]
    top5 = [(labels[i], float(probs[i])) for i in top5_idx]
    
    # 5. Print results
    print("Top-5 Predictions:")
    for name, prob in top5:
        print(f"  {name}: {prob*100:.2f}%")



    url = "http://localhost:8080/predict_resnet50"
    payload = {"data": batch.tolist()}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    
    # 3. Parse logits and compute probabilities
    logits = np.array(resp.json(), dtype=np.float32)
    probs = softmax(logits)
    
    # 4. Get top-5 predictions
    top5_idx = probs.argsort()[-5:][::-1]
    top5 = [(labels[i], float(probs[i])) for i in top5_idx]
    
    # 5. Print results
    print("Top-5 Predictions:")
    for name, prob in top5:
        print(f"  {name}: {prob*100:.2f}%")
if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python request.py <path_to_image>")
    #     sys.exit(1)
    img_path = "WhatsApp Image 2025-08-16 at 22.39.06_adfb401f.jpg"
    predict(img_path)


# import requests
# import numpy as np
# from PIL import Image

# # 1. Load and preprocess the image
# img_path = "IMG-20250823-WA0000.jpg"
# img = Image.open(img_path).convert("RGB")
# img = img.resize((224, 224))                       # Triton expects 224×224
# img_data = np.asarray(img, dtype=np.float32)       # shape (224,224,3)
# img_data = img_data.transpose(2, 0, 1)             # to (3,224,224)
# img_data = img_data / 255.0                        # normalize if your model expects [0,1]
# batch = img_data[np.newaxis, ...]                  # shape (1,3,224,224)

# # 2. Send request
# response = requests.post(
#     "http://localhost:8080/predict",
#     json={"data": batch.tolist()},
#     headers={"Content-Type": "application/json"}
# )

# # 3. Parse response
# if response.status_code == 200:
#     preds = np.array(response.json(), dtype=np.float32)
#     top5 = preds.argsort()[-5:][::-1]
#     print("Top-5 class IDs:", top5)
#     print("Top-5 scores:  ", preds[top5])
# else:
#     print("Request failed:", response.status_code, response.text)
