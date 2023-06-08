import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import scipy

parent_dir = r'C:\Projects\Dataset\collected_set'
augmented_dir = r'C:\Projects\Dataset\augmented'

categories = ['Facemask' , 'incorrect' , 'no_face_mask']

datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode = 'reflect'
    
    
)
# to png using PIL
# for category in categories:
#     path = os.path.join(parent_dir , category)
#     for img in os.listdir(path):
#         if img.endswith(".png"):
#             image = Image.open(os.path.join(path , img))

#             new_img_path = os.path.splitext(img)[0]+'.jpg'
#             image.convert('RGB').save(os.path.join(path , new_img_path) , 'JPEG')
#             os.remove(os.path.join(path , img))

#augmentation

for category in categories:
    current_dir = os.path.join(parent_dir , category)
    destination_dir = os.path.join(augmented_dir, category)
    for img in os.listdir(current_dir):
        image = cv2.imread(os.path.join(current_dir , img))
        img_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb , (224,224))
        img_reshape = np.expand_dims(img_resize , axis=0)
        i = 0

        for batch in datagen.flow(img_reshape , batch_size=1 , save_to_dir=destination_dir , save_format='jpg' ):
            i+=1
            print("augmenting.")
            if i>16:
                break



        



