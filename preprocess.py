from net import Net
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import pandas as pd
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split

PATH = '../datasets/TBX11K/TBX11K/imgs/'
train_txt = '../datasets/TBX11K/TBX11K/lists/TBX11K_trainval.txt'

filenames = None

req_filenames = []
categories = []
with open(train_txt) as f:
    filenames = [i.strip() for i in f.readlines()]

for filename in filenames:
    if filename.startswith('tb'):
        categories.append('tb')
        req_filenames.append(filename)
    elif filename.startswith('health'):
        categories.append('health')
        req_filenames.append(filename)
    else:
        continue

df = pd.DataFrame({
    'filename': req_filenames,
    'category': categories
    })

df = df.sample(frac=1)


train_df, validate_df = train_test_split(df, test_size=0.2)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1.0/255.0,
    horizontal_flip=True)
validate_datagen = ImageDataGenerator(rescale=1.0/255.0)


validate_generator = validate_datagen.flow_from_dataframe(
    validate_df, 
    PATH,
    x_col = 'filename',
    y_col = 'category',
    target_size=(512, 512),
    class_mode='categorical',
    batch_size=15
    )


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    PATH,
    x_col = 'filename',
    y_col = 'category',
    target_size = (512, 512),
    class_mode='categorical',
    batch_size=15
    )

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    PATH,
    x_col='filename',
    y_col='category',
    target_size=(512, 512),
    class_mode='categorical'
    )

plt.figure(figsize=(12, 12))
for i in range(15):
    plt.subplot(5, 3, i+1)
    for x, y in example_generator:
        image = x[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
