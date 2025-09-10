import json
import numpy as np
import cv2
import base64
import io
from PIL import Image
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """
    Preprocessing model for image data
    Handles raw image bytes, base64 encoded images, or numpy arrays
    """
    
    def initialize(self, args):
        """Initialize preprocessing parameters"""
        print("Initializing preprocessing model...")
        
        # Parse model configuration
        self.model_config = json.loads(args['model_config'])
        
        # Image preprocessing parameters
        self.target_height = 224
        self.target_width = 224
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print("Preprocessing model initialized successfully")
        
    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_IMAGE")
                input_data = input_tensor.as_numpy()  # (N,) 1D uint8 bytes array
                
                processed_image = self._preprocess_image(input_data)  # shape (3,224,224)
                
                # Add batch dimension: (1, 3, 224, 224)
                output_data = np.expand_dims(processed_image, axis=0).astype(np.float32)
                
                print(f"Preprocess output shape: {output_data.shape}, dtype: {output_data.dtype}")  # debug
                
                output_tensor = pb_utils.Tensor("data", output_data)
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)
            except Exception as e:
                print(f"Preprocess error: {str(e)}")
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(e))
                    )
                )
        return responses


    
    def _preprocess_image(self, image_data):
        """Preprocess a single image"""
        try:
            # Decode image from bytes
            # image_bytes = image_data.tobytes()
            # image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            
            # Decode using OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (self.target_width, self.target_height))
            
            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            image = (image - self.mean) / self.std
            
            # Convert from HWC to CHW format
            image = np.transpose(image, (2, 0, 1))
            
            return image
            
        except Exception as e:
            print(f"Error in _preprocess_image: {e}")
            # Return a black image if preprocessing fails
            return np.zeros((3, self.target_height, self.target_width), dtype=np.float32)
    
    def finalize(self):
        """Cleanup resources"""
        print("Cleaning up preprocessing model")
