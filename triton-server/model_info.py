# model_info.py - Helper to inspect your ONNX model
import onnx
import sys

def inspect_onnx_model(model_path):
    """Inspect ONNX model to get input/output information"""
    try:
        model = onnx.load(model_path)
        
        print(f"Model: {model_path}")
        print("=" * 50)
        
        # Get model inputs
        print("INPUTS:")
        for input_tensor in model.graph.input:
            name = input_tensor.name
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            
            dtype = input_tensor.type.tensor_type.elem_type
            dtype_map = {1: "FLOAT", 2: "UINT8", 3: "INT8", 6: "INT32", 7: "INT64"}
            dtype_str = dtype_map.get(dtype, f"TYPE_{dtype}")
            
            print(f"  Name: {name}")
            print(f"  Shape: {shape}")
            print(f"  Type: {dtype_str}")
            print()
        
        # Get model outputs  
        print("OUTPUTS:")
        for output_tensor in model.graph.output:
            name = output_tensor.name
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            
            dtype = output_tensor.type.tensor_type.elem_type
            dtype_map = {1: "FLOAT", 2: "UINT8", 3: "INT8", 6: "INT32", 7: "INT64"}
            dtype_str = dtype_map.get(dtype, f"TYPE_{dtype}")
            
            print(f"  Name: {name}")
            print(f"  Shape: {shape}")
            print(f"  Type: {dtype_str}")
            print()
        
        # Generate config template
        print("SUGGESTED config.pbtxt:")
        print("=" * 50)
        
        # Get first input/output for template
        first_input = model.graph.input[0]
        first_output = model.graph.output[0]
        
        input_name = first_input.name
        input_shape = []
        for dim in first_input.type.tensor_type.shape.dim:
            if dim.dim_value:
                input_shape.append(dim.dim_value)
            else:
                input_shape.append(-1)
        
        output_name = first_output.name
        output_shape = []
        for dim in first_output.type.tensor_type.shape.dim:
            if dim.dim_value:
                output_shape.append(dim.dim_value)
            else:
                output_shape.append(-1)
        
        # Remove batch dimension (first dimension) for config
        if input_shape and input_shape[0] == -1:
            input_shape = input_shape[1:]
        if output_shape and output_shape[0] == -1:
            output_shape = output_shape[1:]
        
        config_template = f'''name: "my_onnx_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
default_model_filename: "model.onnx"

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: {input_shape}
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: {output_shape}
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100
}}'''

        print(config_template)
        
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_info.py <path_to_onnx_model>")
        sys.exit(1)
    
    inspect_onnx_model(sys.argv[1])
