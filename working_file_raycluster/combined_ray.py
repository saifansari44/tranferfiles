import requests
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from ray import serve

app = FastAPI()

# Load ImageNet class names once for prediction labels
with open("synset.txt") as f:
    labels = [line.strip() for line in f]

def preprocess_image_bytes(image_bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize [0,1]
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    return arr[np.newaxis, ...]  # add batch dim

def softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class TritonClient:
    def __init__(self):
        self.triton_urls = {
            "squeezenet_onnx": {
                "url": "http://triton-server.rayserve.svc.cluster.local:8000/v2/models/squeezenet_onnx/infer",
                "output": "squeezenet0_flatten0_reshape0"
            },
            "resnet50_onnx": {
                "url": "http://triton-server.rayserve.svc.cluster.local:8000/v2/models/resnet50_onnx/infer",
                "output": "resnetv17_dense0_fwd"
            }
        }

    @app.post("/predict")
    async def predict(self, image: UploadFile = File(...), model: str = Form("squeezenet_onnx")):
        image_bytes = await image.read()
        np_input = preprocess_image_bytes(image_bytes)
        
        if model not in self.triton_urls:
            return {"error": f"Model '{model}' not supported."}
        
        triton_info = self.triton_urls[model]

        payload = {
            "inputs": [
                {
                    "name": "data",
                    "shape": np_input.shape,
                    "datatype": "FP32",
                    "data": np_input.flatten().tolist(),
                }
            ],
            "outputs": [{"name": triton_info["output"]}]
        }

        response = requests.post(triton_info["url"], json=payload)
        if response.status_code != 200:
            return {"error": "Triton inference request failed."}
        
        result = response.json()
        logits = np.array(result["outputs"][0]["data"], dtype=np.float32)
        probs = softmax(logits)

        top5_idx = probs.argsort()[-5:][::-1]
        top5 = [(labels[i], float(probs[i])) for i in top5_idx]

        return {"predictions": top5}

    @app.get("/health")
    async def health(self):
        # Check if we can connect to any Triton server
        try:
            response = requests.get("http://triton-server.rayserve.svc.cluster.local:8000/v2/health/ready")
            return {"status": "healthy" if response.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @app.get("/")
    async def root(self):
        return JSONResponse({
            "service": "Ray→Triton Image Service",
            "endpoints": {
                "predict": "POST /predict - Upload image for classification",
                "health": "GET /health - Check service health", 
                "docs": "GET /docs - API documentation"
            },
            "triton_url": self.triton_url,
            "model_name": self.model_name
        })

if __name__ == "__main__":
    import ray
    import time

    ray.init()
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    handle = TritonClient.bind()
    serve.run(handle)
    print("Serve is ready and running at port 8000")

    # Keep the script alive
    while True:
        time.sleep(60)
