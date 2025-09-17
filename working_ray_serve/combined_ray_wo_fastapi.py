import tempfile
import json
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from ray import serve

@serve.deployment()
class TritonClient:
    def __init__(self):
        # Load ImageNet class names once for prediction labels
        try:
            with open("synset.txt") as f:
                self.labels = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print("Warning: synset.txt not found. Labels will not be available.")
            self.labels = []
        # to run in cluster, use the internal k8s service name
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

        # To run locally 
        # self.triton_urls = {
        #     "squeezenet_onnx": {
        #         "url": "http://localhost:8000/v2/models/squeezenet_onnx/infer",
        #         "output": "squeezenet0_flatten0_reshape0"
        #     },
        #     "resnet50_onnx": {
        #         "url": "http://localhost:8000/v2/models/resnet50_onnx/infer",
        #         "output": "resnetv17_dense0_fwd"
        #     }
        # }
        print("TritonClient initialized successfully!")
    
    def preprocess_image_bytes(self, image_bytes) -> np.ndarray:
        """Preprocess image bytes to the format expected by the models"""
        img = Image.open(BytesIO(image_bytes))
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC to CHW
        return arr[np.newaxis, ...]  # add batch dim
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities"""
        exps = np.exp(logits - np.max(logits))
        return exps / exps.sum()
    
    async def __call__(self, http_request):
        """Handle HTTP requests for image classification"""
        try:
            # Parse the request path to determine the endpoint
            path = http_request.url.path
            method = http_request.method
            
            if method == "GET" and path == "/":
                return await self.root()
            elif method == "GET" and path == "/health":
                return await self.health()
            elif method == "POST" and path == "/predict":
                return await self.predict(http_request)
            else:
                return {"error": f"Endpoint {method} {path} not found"}
                
        except Exception as e:
            return {"error": f"Request processing failed: {str(e)}"}
    
    async def predict(self, http_request):
        """Handle image prediction requests"""
        try:
            # Parse form data from the request
            request_form = await http_request.form()
            
            # Get the image file
            if "image" not in request_form:
                return {"error": "No image file provided"}
            
            image_file = request_form["image"]
            image_bytes = await image_file.read()
            
            # Get the model parameter (default to squeezenet_onnx)
            model = request_form.get("model", "squeezenet_onnx")
            
            # Preprocess the image
            np_input = self.preprocess_image_bytes(image_bytes)
            
            # Check if model is supported
            if model not in self.triton_urls:
                return {"error": f"Model '{model}' not supported."}
            
            # Prepare Triton inference request
            triton_info = self.triton_urls[model]
            payload = {
                "inputs": [
                    {
                        "name": "data",
                        "shape": list(np_input.shape),
                        "datatype": "FP32",
                        "data": np_input.flatten().tolist(),
                    }
                ],
                "outputs": [{"name": triton_info["output"]}]
            }
            
            # Send request to Triton server
            response = requests.post(triton_info["url"], json=payload)
            if response.status_code != 200:
                return {"error": "Triton inference request failed."}
            
            # Process the response
            result = response.json()
            logits = np.array(result["outputs"][0]["data"], dtype=np.float32)
            probs = self.softmax(logits)
            
            # Get top 5 predictions
            top5_idx = probs.argsort()[-5:][::-1]
            if self.labels:
                top5 = [(self.labels[i], float(probs[i])) for i in top5_idx]
            else:
                top5 = [(f"class_{i}", float(probs[i])) for i in top5_idx]
            
            return {"predictions": top5}
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    async def health(self):
        """Check health of Triton server"""
        try:
            response = requests.get("http://triton-server.rayserve.svc.cluster.local:8000/v2/health/ready")
            return {"status": "healthy" if response.status_code == 200 else "unhealthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def root(self):
        """Root endpoint with service information"""
        return {
            "service": "Ray→Triton Image Service",
            "endpoints": {
                "predict": "POST /predict - Upload image for classification",
                "health": "GET /health - Check service health"
            },
            "supported_models": list(self.triton_urls.keys()),
            "usage": {
                "predict": "Send POST request to /predict with 'image' file and optional 'model' parameter",
                "models": list(self.triton_urls.keys())
            }
        }

# Create the deployment
app = TritonClient.bind()

# if __name__ == "__main__":
#     import ray
#     import time

#     ray.init()
#     serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8100})
#     handle = TritonClient.bind()
#     serve.run(handle)
#     print("Serve is ready and running at port 8100")

#     # Keep the script alive
#     while True:
#         time.sleep(60)