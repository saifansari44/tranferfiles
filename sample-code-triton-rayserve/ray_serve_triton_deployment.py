import requests
import numpy as np
from ray import serve
from fastapi import FastAPI
import time
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    data: list  # expects [batch, 3, 224, 224]

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class TritonClient:
    def __init__(self):
        self.triton_url_squeezenet = "http://localhost:8000/v2/models/squeezenet_onnx/infer"
        self.triton_url_resnet50 = "http://localhost:8000/v2/models/resnet50_onnx/infer"

    @app.post("/predict_squeezenet")
    async def predict(self, req: InferenceRequest):
        np_input = np.array(req.data, dtype=np.float32)
        payload = {
            "inputs": [
                {
                    "name": "data",
                    "shape": np_input.shape,
                    "datatype": "FP32",
                    "data": np_input.flatten().tolist(),
                }
            ],
            "outputs": [{"name": "squeezenet0_flatten0_reshape0"}]
        }
        response = requests.post(self.triton_url_squeezenet, json=payload)
        result = response.json()
        print(result)
        return result["outputs"][0]["data"]
    
    @app.post("/predict_resnet50")
    async def predict(self, req: InferenceRequest):
        np_input = np.array(req.data, dtype=np.float32)
        payload = {
            "inputs": [
                {
                    "name": "data",
                    "shape": np_input.shape,
                    "datatype": "FP32",
                    "data": np_input.flatten().tolist(),
                }
            ],
            "outputs": [{"name": "resnetv17_dense0_fwd"}]
        }
        response = requests.post(self.triton_url_resnet50, json=payload)
        result = response.json()
        print(result)
        return result["outputs"][0]["data"]

# Initialize Ray and Serve
if __name__ == "__main__":
    import ray
    ray.init()
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8080})
    # TritonClient.deploy()
    handle = TritonClient.bind()
    serve.run(handle)  
    print("Serve is ready; entering blocking loop")
    # Keep the script alive
    while True:
        time.sleep(60)