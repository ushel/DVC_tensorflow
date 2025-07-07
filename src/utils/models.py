import tensorflow as tf  
import os  
import joblib   
import logging


def get_VGG_16_model(input_shape,model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False  # getting rid of fully connected layer of VGG16 model we just keeping onvolution layer which is trained on imagenet dataset
    )
    
    model.save(model_path)
    logging.info(f"VGG16 model saved at {model_path}")
    return model

def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False 
    elif(freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False 
            
    # add our own fully connected layers... 
    # 7 X 7 X 512 to single array it can be process as an input layer for fully connected layer
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense( #as tis is other layer need to connect with flatten so think it as function
        units = CLASSES,
        activation="softmax"
    )(flatten_in) # functional approach
    
    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )
    
    # logging.info("custom model summary")
    full_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )
    
    logging.info("custom model is compiled and ready to be trained.")
    
    return full_model

def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from {untrained_full_model_path}")
    return model