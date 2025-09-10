import tritonclient.http as httpclient
import numpy as np

def infer_image(image_path, server_url="localhost:8000", model_name="ensemble_model"):
    # Read raw image bytes
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    # Convert raw bytes to a numpy array with batch dim
    # input_data = np.array([np.frombuffer(raw_bytes, dtype=np.uint8)], dtype=np.uint8)
    
    # Create Triton client
    client = httpclient.InferenceServerClient(url=server_url)
    
    # Define input tensor - batch size 1, dims = [-1]
    inputs = []
    # inputs.append(httpclient.InferInput(name="RAW_IMAGE", shape=input_data.shape, datatype="UINT8"))
    # inputs[0].set_data_from_numpy(input_data)


    input_data = np.frombuffer(raw_bytes, dtype=np.uint8)  # shape (N,)
    inputs.append(httpclient.InferInput(name="RAW_IMAGE", shape=input_data.shape, datatype="UINT8"))
    inputs[0].set_data_from_numpy(input_data)
        
    # Define outputs to retrieve
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("TOP_CLASSES"))
    outputs.append(httpclient.InferRequestedOutput("TOP_PROBABILITIES"))
    outputs.append(httpclient.InferRequestedOutput("TOP_LABELS"))
    
    # Perform inference
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    # Extract results
    top_classes = response.as_numpy("TOP_CLASSES")[0]
    top_probs = response.as_numpy("TOP_PROBABILITIES")[0]
    top_labels = response.as_numpy("TOP_LABELS")[0]
    
    # Decode labels if needed (assuming INT8 bytes, convert accordingly)
    labels = [label.tobytes().decode("utf-8") if hasattr(label, "tobytes") else str(label) for label in top_labels]
    
    print("Top predictions:")
    for cls, prob, label in zip(top_classes, top_probs, labels):
        print(f"Class {cls} ({label}): {prob:.4f}")

# Example usage
infer_image("cat.jpg")
