import onnx
from onnx import numpy_helper
from onnx import TensorProto

def inspect_onnx_model(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    # Graph inputs
    print("=== Graph Inputs ===")
    for inp in graph.input:
        # Build a readable shape, using '?' for unknown dims
        shape = [dim.dim_value if dim.dim_value > 0 else '?' 
                 for dim in inp.type.tensor_type.shape.dim]
        # Get the element type enum value
        dtype_enum = inp.type.tensor_type.elem_type
        # Convert enum to string name, e.g. FLOAT, INT64
        dtype_name = TensorProto.DataType.Name(dtype_enum)
        print(f"Name: {inp.name}, Type: {dtype_name}, Shape: {shape}")
    print()

    # Graph outputs
    print("=== Graph Outputs ===")
    for out in graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else '?' 
                 for dim in out.type.tensor_type.shape.dim]
        dtype_enum = out.type.tensor_type.elem_type
        dtype_name = TensorProto.DataType.Name(dtype_enum)
        print(f"Name: {out.name}, Type: {dtype_name}, Shape: {shape}")
    print()

    # Initializers (weights)
    print("=== Initializers ===")
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        print(f"Name: {init.name}, Dtype: {arr.dtype}, Shape: {arr.shape}")
    print()

    # Intermediate values
    print("=== ValueInfo (Intermediate Tensors) ===")
    for val in graph.value_info:
        shape = [dim.dim_value if dim.dim_value > 0 else '?' 
                 for dim in val.type.tensor_type.shape.dim]
        dtype_enum = val.type.tensor_type.elem_type
        dtype_name = TensorProto.DataType.Name(dtype_enum)
        print(f"Name: {val.name}, Type: {dtype_name}, Shape: {shape}")
    print()

    # Nodes
    print("=== Graph Nodes (Operators) ===")
    for node in graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) 
                 for a in node.attribute}
        print(f"OpType: {node.op_type}, Name: {node.name or '<unnamed>'}")
        print(f"  Inputs : {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        if attrs:
            print(f"  Attributes: {attrs}")
    print()

if __name__ == "__main__":
    # inspect_onnx_model("model_repository/resnet50_onnx/1/model.onnx")
    inspect_onnx_model("model_repository/squeezenet_onnx/1/model.onnx")
    
