import tensorflow as tf  
import os
# import time 
import joblib   # helps us to store information as binary
import logging
from src.utils.all_utils import get_timestamp


# def get_timestamp(name):
#     timestamp = time.asctime().replace(" ","_").replace(":","_")
#     unique_name = f"{name}_at_{timestamp}"
#     return unique_name
    
    

def create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir):
    unique_name = get_timestamp("tb_logs")
    
    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    tb_callback_filepath = os.path.join(callbacks_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callbacks, tb_callback_filepath)
    logging.info(f"Tensorboard callback is being saved at {tb_callback_filepath}")

def create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir):
    checkpoint_file_path = os.path.join(checkpoint_dir,"ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_best_only=True, # if set False will save for all epochs
    )
    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)
    logging.info(f"Checkpoint callback is being saved at {ckpt_callback_filepath}")