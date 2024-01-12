import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define hyperparameters
batch_size = 32
input_shape = (32, 32, 3)  # Adjust the input shape as needed

# No data augmentation for test set
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
data_folder = 'non_face_detector_samp_1/'

# Create a test data generator
test_generator = test_datagen.flow_from_directory(
    data_folder + 'Test',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Load the trained model
model = tf.keras.models.load_model('best_face_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')
