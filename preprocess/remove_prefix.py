import glob
import re

import os

# move all ./dataset/image_*(.*).jpg to ./dataset/$1.jpg exclude the image_*
def move_images():
    # Get all image files in the dataset directory
    image_files = glob.glob('./dataset/image_*.jpg')
    
    for file_path in image_files:
        # Extract the new filename by removing 'image_\d+' prefix
        basename = os.path.basename(file_path)
        new_filename = re.sub(r'image_\d+_(.*)', r'\1', basename)
        
        # Define the new path
        new_path = os.path.join('./dataset', new_filename)
        
        # Move the file
        os.rename(file_path, new_path)
        print(f'Moved {file_path} to {new_path}')

if __name__ == "__main__":
    move_images()
    print("All images moved successfully.")