from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import get_VGG_16_model,prepare_model
import argparse 
import pandas as pd  
import os   
import shutil
from tqdm import tqdm  # print progress bar
import logging
import io 


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
# create_directory([log_dir]) # not storing logs so use os.makdirs
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), level = logging.INFO, format=logging_str, filemode='a')  
    
def prepare_base_model(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # 2. Use transfer learning
    artifacts = config["artifacts"]
    artifacts_dir= artifacts["ARTIFACTS_DIR"]
    
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]
    base_model_dir_path = os.path.join(artifacts_dir,base_model_dir)
    
    create_directory([base_model_dir_path])
    
    base_model_path = os.path.join(base_model_dir_path,base_model_name)
  
    model = get_VGG_16_model(input_shape=params["IMAGE_SIZE"], model_path = base_model_path)
    full_model  = prepare_model(
                model,
                CLASSES=params["CLASSES"],
                freeze_all=True,
                freeze_till=None,
                learning_rate=params["LEARNING_RATE"]
                )
    update_base_model_path = os.path.join(
        base_model_dir_path,
        artifacts["UPDATED_BASE_MODEL_NAME"]
    )
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str
    logging.info(f"full model summary: \n{_log_model_summary(full_model)}")
    
    # logging.info(f"{full_model.summary()}") # it printing in terminal and not in logs i want it in logs...we need to do some customization
    full_model.save(update_base_model_path)
    
    #transfer learning means we will only remove fully connected layers and will keep all the convolutional layers whihci is trained on imagenet dataset, also fully connected layers has 1000 classes we only required 2 classes.
    
    # 3. Prepare the callbacks tensorboard logs or checkpointing
    
    # 4. Training module(take data from get data section, prepare custom model and take all the call back ---> trained model (file required  - config.yaml, secrets.yaml, params.yaml, artifacts directory,dvc.yaml,loogs.yaml, setup.py))
    # artifacts folder --> model, callbacks binary 
    # logs --> General logs and tensorboard logs 

        
 from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import get_VGG_16_model,prepare_model
from src.utils.callbacks import create_and_save_tensorboard_callback,create_and_save_checkpoint_callback
import argparse 
import pandas as pd  
import os   
import shutil
from tqdm import tqdm  # print progress bar
import logging
import io 


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
# create_directory([log_dir]) # not storing logs so use os.makdirs
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"), level = logging.INFO, format=logging_str, filemode='a')  
    
def prepare_base_model(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # 2. Use transfer learning
    artifacts = config["artifacts"]
    artifacts_dir= artifacts["ARTIFACTS_DIR"]
    
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]
    base_model_dir_path = os.path.join(artifacts_dir,base_model_dir)
    
    create_directory([base_model_dir_path])
    
    base_model_path = os.path.join(base_model_dir_path,base_model_name)
  
    model = get_VGG_16_model(input_shape=params["IMAGE_SIZE"], model_path = base_model_path)
    full_model  = prepare_model(
                model,
                CLASSES=params["CLASSES"],
                freeze_all=True,
                freeze_till=None,
                learning_rate=params["LEARNING_RATE"]
                )
    update_base_model_path = os.path.join(
        base_model_dir_path,
        artifacts["UPDATED_BASE_MODEL_NAME"]
    )
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str
    logging.info(f"full model summary: \n{_log_model_summary(full_model)}")
    
    # logging.info(f"{full_model.summary()}") # it printing in terminal and not in logs i want it in logs...we need to do some customization
    full_model.save(update_base_model_path)
    
    #transfer learning means we will only remove fully connected layers and will keep all the convolutional layers whihci is trained on imagenet dataset, also fully connected layers has 1000 classes we only required 2 classes.
    
    # 3. Prepare the callbacks tensorboard logs or checkpointing
    
    # 4. Training module(take data from get data section, prepare custom model and take all the call back ---> trained model (file required  - config.yaml, secrets.yaml, params.yaml, artifacts directory,dvc.yaml,loogs.yaml, setup.py))
    # artifacts folder --> model, callbacks binary 
    # logs --> General logs and tensorboard logs 
def prepare_callbacks(config_path, params_path):
    config = read_yaml(config_path)
    parmas = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = config["ARTIFACTS_DIR"]
    tensorboard_log_dir = os.pat.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])
        
    checkpoint_dir= os.path.join(artifacts_dir,artifacts["CHECKPOINT_DIR"])
    callbacks_dir= os.path.join(artifacts_dir,artifacts["CALLBACKS_DIR"])
    
    create_directory([
        tensorboard_log_dir,
        checkpoint_dir,
        callbacks_dir
    ])
    
    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir)    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    args.add_argument("--config","-c",default="config/config.yaml")
    
    args.add_argument("--params","-p",default="params.yaml")

    
    parsed_args = args.parse_args()
    
    try:
        # print_fn("test") # to check exception
        logging.info("\n>>>>>>>>>>>>>>>>stage three started")
        prepare_callbacks(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage three completed! callbacks are prepared and saved as binary >>>>>>>>>>>>>>>>>>>>")
        
    except Exception as e:
        logging.exception(e)
        raise e    