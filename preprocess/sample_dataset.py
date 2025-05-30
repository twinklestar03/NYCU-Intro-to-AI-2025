import glob
import random
import os

DEST_FOLDER = './dataset'

# choose man
man_images = glob.glob(f'./images/MEN-*')
women_images = glob.glob(f'./images/WOMEN-*')

# randomly select 500 images from each category
man_selected = random.sample(man_images, 500)
women_selected = random.sample(women_images, 500)

joined_images = man_selected + women_selected
random.shuffle(joined_images)

# copy selected images to destination folder
for i, image in enumerate(joined_images):
    image_name = image.split('/')[-1]
    dest_path = f'{DEST_FOLDER}/{image_name}'
    os.system(f'cp {image} {DEST_FOLDER}/image_{i}_{image_name}')
