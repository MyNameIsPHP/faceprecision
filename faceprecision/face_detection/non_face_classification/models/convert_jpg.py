from PIL import Image
import os

# Set the folder path containing the images
folder_path = 'non_face_detector/not_face'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpeg', '.gif', '.bmp', '.tiff')):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Open the image
        with Image.open(file_path) as img:
            # Convert the image to RGB mode (necessary for .jpg)
            rgb_img = img.convert('RGB')
            # Define the new filename, replacing the old extension with .jpg
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            new_file_path = os.path.join(folder_path, new_filename)
            # Save the image in JPG format
            rgb_img.save(new_file_path, 'JPEG')

print("Conversion complete.")
