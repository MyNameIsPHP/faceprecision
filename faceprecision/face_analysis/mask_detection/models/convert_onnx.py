import argparse
import tensorflow as tf
import tf2onnx

# Initialize argument parser
parser = argparse.ArgumentParser(description='Convert a Keras model to ONNX format.')
parser.add_argument('input_model', help='Path to the input Keras model file (e.g., model.h5)')

# Parse arguments
args = parser.parse_args()

# Load the trained model
model = tf.keras.models.load_model(args.input_model, compile=False)

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11, output_path=args.input_model.split(".")[0] + ".onnx")

# Print output names
output_names = [n.name for n in onnx_model.graph.output]
print("Output names: ", output_names)
