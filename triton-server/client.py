# simple_client.py
import tritonclient.http as httpclient
import numpy as np
import cv2
import time
import requests

with open("synset.txt") as f:
    # labels = [line.strip() for line in f]
    labels = [line.strip() for line in f.readlines()]
if len(labels) != 1000:
    raise ValueError(f"Expected 1000 labels, got {len(labels)}")


# print(labels)

class SimpleONNXClient:
    """Simple client for ONNX model inference"""
    
    def __init__(self, server_url="localhost:8000", model_name="my_onnx_model"):
        self.server_url = server_url
        self.model_name = model_name
        self.client = None
        
    def connect(self):
        """Connect to Triton server"""
        try:
            self.client = httpclient.InferenceServerClient(url=self.server_url)
            
            # Check if server and model are ready
            if not self.client.is_server_live():
                print("Error: Server is not live")
                return False
                
            if not self.client.is_model_ready(self.model_name):
                print(f"Error: Model {self.model_name} is not ready")
                return False
                
            print(f"✓ Connected to {self.server_url}")
            print(f"✓ Model {self.model_name} is ready")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Simple image preprocessing"""
        try:
            # Load and resize image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            
            # Normalize to [0, 1] and convert to CHW format
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def infer(self, input_data):
        """Perform inference"""
        try:
            start_time = time.time()
            
            # Prepare input
            inputs = []
            inputs.append(httpclient.InferInput("data", input_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_data)
            
            # Prepare output
            outputs = []
            if self.model_name == "squeezenet_onnx":
                outputs.append(httpclient.InferRequestedOutput("squeezenet0_flatten0_reshape0"))
            if self.model_name == "resnet50_onnx":
                outputs.append(httpclient.InferRequestedOutput("resnetv17_dense0_fwd"))
            
            # Run inference
            results = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Get results
            if self.model_name == "squeezenet_onnx":
                output_data = results.as_numpy("squeezenet0_flatten0_reshape0")
            elif self.model_name == "resnet50_onnx":
                output_data = results.as_numpy("resnetv17_dense0_fwd")
            # output_data = results.as_numpy("squeezenet0_flatten0_reshape0")
            inference_time = time.time() - start_time
            
            print(f"Inference completed in {inference_time:.4f}s")
            print(f"Output shape: {output_data.shape}")
            
            return output_data
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def infer_image(self, image_path, top_k=5):
        """End-to-end image inference"""
        # Preprocess
        input_data = self.preprocess_image(image_path)
        if input_data is None:
            return None
        
        # Infer
        output = self.infer(input_data)
        if output is None:
            return None
        
        # Simple postprocessing (for classification)
        if len(output.shape) == 2 and output.shape[1] > 1:
            # Classification case
            probabilities = output[0]
            # Apply softmax if logits
            if probabilities.max() > 1.0:                # CHANGED: detect logits vs probs
                exp = np.exp(probabilities - probabilities.max())
                probabilities = exp / exp.sum()

             # Get top-k
            top_indices = np.argsort(probabilities)[::-1][:top_k]   # CHANGED: use top_k
            print(f"Top {top_k} predictions:")
            for rank, idx in enumerate(top_indices, start=1):
                label = labels[idx]           # CHANGED: map index to label
                score = probabilities[idx]
                print(f"  {rank}. {label} (class {idx}): {score:.4f}")  # CHANGED: print label

            # predicted_class = np.argmax(probabilities)
            # confidence = probabilities[predicted_class]
            
            # print(f"Predicted class: {predicted_class}")
            # print(f"Confidence: {confidence:.4f}")
            
            # # Show top 5 predictions
            # top_indices = np.argsort(probabilities)[::-1][:5]
            # print("Top 5 predictions:")
            # for i, idx in enumerate(top_indices):
            #     print(f"  {i+1}. Class {idx}: {probabilities[idx]:.4f}")
        
        return output

def main():
    """Simple usage example"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ONNX Triton Client")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="squeezenet_onnx", help="Model name")
    parser.add_argument("--server", default="localhost:8000", help="Server URL")
    
    args = parser.parse_args()
    
    # requests.post(f"http://localhost:8000/v2/repository/load/{args.model}")

    # Create and connect client
    client = SimpleONNXClient(server_url=args.server, model_name=args.model)
    
    if not client.connect():
        return
    
    # Run inference
    print(f"\nRunning inference on: {args.image}")
    result = client.infer_image(args.image)
    
    if result is not None:
        print("✓ Inference successful")
    else:
        print("✗ Inference failed")

if __name__ == "__main__":
    main()
