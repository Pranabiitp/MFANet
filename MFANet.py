#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def conv_block(x, filters, kernel_size, strides, padding='same', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = layers.BatchNormalization()(x)
    return x

def channel_attention_block(x, reduction_ratio=8):
    channel = x.shape[-1]
    shared_dense_one = layers.Dense(channel // reduction_ratio, activation='relu')
    shared_dense_two = layers.Dense(channel, activation='sigmoid')

    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attn = layers.Add()([avg_pool, max_pool])
    attn = layers.Activation('sigmoid')(attn)

    return layers.Multiply()([x, attn])

def spatial_attention_block(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attn = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, attn])

def cross_scale_interaction_attention(fibers):
    min_channels = min([fiber.shape[-1] for fiber in fibers])

    fused_features = []
    for i, fiber in enumerate(fibers):
        fiber = layers.Conv2D(min_channels, kernel_size=1, padding='same')(fiber)
        
        channel_attn = channel_attention_block(fiber)
        spatial_attn = spatial_attention_block(fiber)
        interaction = 0

        for j, other_fiber in enumerate(fibers):
            if i != j:
                other_fiber = layers.Conv2D(min_channels, kernel_size=1, padding='same')(other_fiber)
                interaction += layers.Multiply()([fiber, other_fiber])

        fused = layers.Add()([fiber, channel_attn, spatial_attn, interaction])
        fused_features.append(fused)

    return fused_features

def multi_feature_extraction(input_shape):
    inputs = Input(shape=input_shape)
    fibers = []

    filters_list = [32, 64, 128]
    for filters in filters_list:
        x = inputs
        for _ in range(3):
            x = conv_block(x, filters, kernel_size=3, strides=1)
            x = layers.MaxPooling2D(pool_size=2)(x)
        fibers.append(x)

    return Model(inputs, fibers, name='MultiFiberFeatureExtraction')

def cross_attention_block(query, key, value, num_heads=8, key_dim=64):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(query, key, value)
    return attn_output

def multi_head_cross_attention_classifier(input_shape, num_classes, num_heads=4):
    inputs = Input(shape=input_shape)
    x = inputs
    attention_heads = []

    query = x
    key = layers.Conv2D(input_shape[-1], kernel_size=3, padding='same')(x)
    value = layers.Conv2D(input_shape[-1], kernel_size=3, padding='same')(x)

    for _ in range(num_heads):
        attn = cross_attention_block(query, key, value, num_heads=8, key_dim=input_shape[-1])
        attention_heads.append(attn)

    concatenated = layers.Concatenate()(attention_heads)
    x = layers.GlobalAveragePooling2D()(concatenated)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='MultiHeadCrossAttentionClassifier')

def build_mfanet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    multi_fiber_extractor = multi_feature_extraction(input_shape)
    fibers = multi_fiber_extractor(inputs)

    fused_features = cross_scale_interaction_attention(fibers)

    min_channels = min([feature.shape[-1] for feature in fused_features])
    fused_features = [layers.Conv2D(min_channels, kernel_size=1, padding='same')(feature) for feature in fused_features]

    concatenated_fused_features = layers.Concatenate()(fused_features)

    classifier = multi_head_cross_attention_classifier(concatenated_fused_features.shape[1:], num_classes)
    outputs = classifier(concatenated_fused_features)

    return Model(inputs, outputs, name='MFAN')

# Example usage
input_shape = (224, 224, 3)
num_classes = 5 #change according to dataset

mfanet_model = build_mfanet(input_shape, num_classes)

