import tensorflow as tf
import h5py
import numpy as np
import os

def explore_h5_structure(file_path):
    """Explore and print the structure of an H5 file"""
    print(f"Exploring H5 file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("\nH5 file keys:", list(f.keys()))
            
            # Print full structure
            print("\nDetailed H5 structure:")
            def print_attrs(name, obj):
                print(f"  {name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    Shape: {obj.shape}, Type: {obj.dtype}")
                for key, val in obj.attrs.items():
                    print(f"    Attr - {key}: {val}")
            
            f.visititems(print_attrs)
            
            # Check for dense layers specifically
            print("\nSearching for dense layers...")
            dense_layers = []
            
            def find_dense_layers(name, obj):
                if 'dense' in name.lower() and isinstance(obj, h5py.Group):
                    dense_layers.append(name)
            
            f.visititems(find_dense_layers)
            print(f"Found dense layers: {dense_layers}")
            
            # Extract weights and biases from sequential_23 (based on the known structure)
            if 'model_weights' in f and 'sequential_23' in f['model_weights']:
                seq = f['model_weights']['sequential_23']
                print("\nExtracting weights from sequential_23:")
                
                if 'dense_Dense11' in seq:
                    dense11 = seq['dense_Dense11']
                    if 'kernel:0' in dense11 and 'bias:0' in dense11:
                        kernel = dense11['kernel:0'][()]
                        bias = dense11['bias:0'][()]
                        print(f"  dense_Dense11/kernel:0 shape: {kernel.shape}")
                        print(f"  dense_Dense11/bias:0 shape: {bias.shape}")
                
                if 'dense_Dense12' in seq:
                    dense12 = seq['dense_Dense12']
                    if 'kernel:0' in dense12:
                        kernel = dense12['kernel:0'][()]
                        print(f"  dense_Dense12/kernel:0 shape: {kernel.shape}")
            
            return True
            
    except Exception as e:
        print(f"Error exploring H5 file: {e}")
        return False

def convert_h5_to_tf_model(file_path, output_path=None):
    """Convert an H5 file to a TensorFlow model by manually loading weights"""
    print(f"Converting H5 file to TensorFlow model: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if model_weights exists
            if 'model_weights' not in f:
                print("Error: model_weights not found in H5 file")
                return None
                
            model_weights = f['model_weights']
            
            # Check for sequential_23 (from the observed structure)
            if 'sequential_23' not in model_weights:
                print("Error: sequential_23 not found in model_weights")
                return None
                
            seq = model_weights['sequential_23']
            
            # Load weights from dense layers
            dense11_weights = None
            dense11_bias = None
            dense12_weights = None
            
            if 'dense_Dense11' in seq:
                dense11 = seq['dense_Dense11']
                if 'kernel:0' in dense11 and 'bias:0' in dense11:
                    dense11_weights = dense11['kernel:0'][()]
                    dense11_bias = dense11['bias:0'][()]
            
            if 'dense_Dense12' in seq:
                dense12 = seq['dense_Dense12']
                if 'kernel:0' in dense12:
                    dense12_weights = dense12['kernel:0'][()]
            
            # Check if we loaded all needed weights
            if dense11_weights is None or dense11_bias is None or dense12_weights is None:
                print("Error: Could not find all required weights and biases")
                return None
                
            # Create a simple TensorFlow model with the same architecture
            input_shape = dense11_weights.shape[0]  # Usually 1280 for EfficientNet backbone
            hidden_units = dense11_weights.shape[1]  # Should be 100
            output_classes = dense12_weights.shape[1]  # Should be 3
            
            print(f"Building model with: input={input_shape}, hidden={hidden_units}, output={output_classes}")
            
            # Define the model architecture - note that the second layer has use_bias=False
            # since there's no bias tensor in the original model for the output layer
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_shape,)),
                tf.keras.layers.Dense(hidden_units, activation='relu', name='dense1'),
                tf.keras.layers.Dense(output_classes, activation='softmax', name='dense2', use_bias=False)
            ])
            
            # Set the weights manually
            model.get_layer('dense1').set_weights([dense11_weights, dense11_bias])
            model.get_layer('dense2').set_weights([dense12_weights])
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Print model summary
            model.summary()
            
            # Save the model if output_path is provided
            if output_path:
                # Ensure it has the correct extension
                if not output_path.endswith('.keras') and not output_path.endswith('.h5'):
                    output_path += '.keras'  # Use .keras extension for TF 2.x
                
                model.save(output_path)
                print(f"Model saved to: {output_path}")
                
            return model
            
    except Exception as e:
        print(f"Error converting H5 file: {e}")
        return None

def test_model_inference(model, input_features=None):
    """Test model inference with random or provided input features"""
    if input_features is None:
        # Generate random features for testing
        input_features = np.random.random((1, model.input_shape[1]))
        print(f"Testing with random input features, shape: {input_features.shape}")
    
    # Run prediction
    predictions = model.predict(input_features, verbose=0)
    print(f"Prediction output shape: {predictions.shape}")
    print(f"Prediction probabilities: {predictions[0]}")
    print(f"Predicted class index: {np.argmax(predictions[0])}")
    
    return predictions

def main():
    # Path to the model file
    model_path = "model_weights/keras_model.h5"
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # First explore the H5 file structure
    if not explore_h5_structure(model_path):
        return
    
    # Convert the H5 file to a TensorFlow model
    model = convert_h5_to_tf_model(model_path, output_path="model_weights/converted_model")
    
    if model:
        # Test the model with random input
        test_model_inference(model)
        
        # Load the labels
        labels_path = "model_weights/labels.txt"
        if os.path.exists(labels_path):
            class_names = open(labels_path, "r").readlines()
            class_names = [name.strip() for name in class_names]
            print(f"\nLoaded {len(class_names)} classes: {class_names}")
            
            # Print class mapping
            for i, name in enumerate(class_names):
                print(f"Class {i}: {name}")
        else:
            print(f"Warning: Labels file not found at {labels_path}")

if __name__ == "__main__":
    main() 