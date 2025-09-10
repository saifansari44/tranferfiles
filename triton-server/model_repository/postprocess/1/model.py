import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """
    Postprocessing model for classification results
    Converts logits to probabilities and returns top-k predictions
    """
    
    def initialize(self, args):
        """Initialize postprocessing parameters"""
        print("Initializing postprocessing model...")
        
        # Parse model configuration
        self.model_config = json.loads(args['model_config'])
        
        # Load class labels
        self.labels = self._load_labels()
        
        # Configuration
        self.top_k = 5
        self.confidence_threshold = 0.001  # Minimum confidence threshold
        
        print(f"Postprocessing model initialized with {len(self.labels)} labels")
        
    def execute(self, requests):
        """Execute postprocessing on model predictions"""
        responses = []
        
        for request in requests:
            try:
                # Get model predictions
                predictions_tensor = pb_utils.get_input_tensor_by_name(request, "MODEL_OUTPUT")
                predictions = predictions_tensor.as_numpy()
                
                batch_size = predictions.shape[0]
                
                # Process each item in the batch
                batch_classes = []
                batch_probabilities = []
                batch_labels = []
                
                for i in range(batch_size):
                    classes, probabilities, labels = self._process_prediction(predictions[i])
                    batch_classes.append(classes)
                    batch_probabilities.append(probabilities)
                    batch_labels.append(labels)
                
                # Convert to numpy arrays
                classes_array = np.array(batch_classes, dtype=np.int32)
                probabilities_array = np.array(batch_probabilities, dtype=np.float32)
                labels_array = np.array(batch_labels, dtype=object)
                
                # Create output tensors
                classes_tensor = pb_utils.Tensor("TOP_CLASSES", classes_array)
                probabilities_tensor = pb_utils.Tensor("TOP_PROBABILITIES", probabilities_array)
                labels_tensor = pb_utils.Tensor("TOP_LABELS", labels_array)
                
                # Create response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[classes_tensor, probabilities_tensor, labels_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                # Handle errors gracefully
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Postprocessing error: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def _process_prediction(self, logits):
        """Process individual prediction"""
        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get top-k indices
        top_indices = np.argsort(probabilities)[::-1][:self.top_k]
        
        # Extract top-k results
        top_classes = top_indices
        top_probabilities = probabilities[top_indices]
        top_labels = [self.labels[idx].encode('utf-8') for idx in top_indices]
        
        return top_classes, top_probabilities, top_labels
    
    def _load_labels(self):
        """Load class labels from file"""
        try:
            # Try to load from synset.txt (adjust path as needed)
            with open("/models/synset.txt", "r") as f:
                labels = [line.strip() for line in f.readlines()]
            
            if len(labels) != 1000:
                print(f"Warning: Expected 1000 labels, got {len(labels)}")
            
            return labels
            
        except Exception as e:
            print(f"Could not load labels file: {e}")
            # Return generic labels if file not found
            return [f"class_{i}" for i in range(1000)]
    
    def finalize(self):
        """Cleanup resources"""
        print("Cleaning up postprocessing model")
