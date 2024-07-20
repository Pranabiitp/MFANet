#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
import numpy as np
def load_img_data(path):
    img_size = (224, 224)
    datagen = ImageDataGenerator()
   
    test_data = datagen.flow_from_directory(
        directory=path,
        target_size=img_size,
        class_mode='categorical',  # Set class_mode to 'categorical' for one-hot encoding
        batch_size=32,
        shuffle=False,  # Set shuffle to False to maintain order

    )

    # Load images and one-hot encode labels into numpy arrays
    images, one_hot_labels = [], []
    for batch in test_data:
        images.extend(batch[0])  # Load images from the batch
        one_hot_labels.extend(batch[1])  # Extract one-hot encoded labels from the batch
        if len(images) >= len(test_data.filenames):
            break

    return np.array(images), np.array(one_hot_labels)

train1,label1=load_img_data('/path/to/the/dataset')
test,label=load_img_data('/path/to/the/dataset')


# In[ ]:


# !pip install imgaug
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Define normal augmentation sequence
def normal_augmentation(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        iaa.Affine(rotate=(-20, 20)),  # rotation
        iaa.Affine(scale=(0.8, 1.2))  # scaling
    ])
    return seq(images=images)

def augment_data(images, labels, augmentation_factor=10):
    augmented_images, augmented_labels = [], []

    for _ in range(augmentation_factor):
        # Apply normal augmentations
        augmented_images_temp = normal_augmentation(images.copy())
        augmented_labels_temp = labels.copy()

        augmented_images.append(augmented_images_temp)
        augmented_labels.append(augmented_labels_temp)

    augmented_images = np.concatenate(augmented_images, axis=0)
    augmented_labels = np.concatenate(augmented_labels, axis=0)

    # Concatenate the augmented data with the original data
    images_final = np.concatenate((images, augmented_images), axis=0)
    labels_final = np.concatenate((labels, augmented_labels), axis=0)

    return images_final, labels_final

train1,label1=augment_data(train1, label1)

