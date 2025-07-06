from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import get_VGG_16_model
import argparse 
import pandas as pd  
import os   
import shutil
from tqdm import tqdm  # print progress bar
import logging


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
    model  = prepare_model(
                model,
                CLASSES=params["CLASSES"],
                freeze_all=True,
                freeze_till=None,
                learning_rate=params["LEARNING_RATE"]
                           )
    
    model = get_VGG_16_model(input_shape=params["IMAGE_SIZE"], model_path = base_model_path)
    #transfer learning means we will only remove fully connected layers and will keep all the convolutional layers whihci is trained on imagenet dataset, also fully connected layers has 1000 classes we only required 2 classes.
    
    # 3. Prepare the callbacks tensorboard logs or checkpointing
    
    # 4. Training module(take data from get data section, prepare custom model and take all the call back ---> trained model (file required  - config.yaml, secrets.yaml, params.yaml, artifacts directory,dvc.yaml,loogs.yaml, setup.py))
    # artifacts folder --> model, callbacks binary 
    # logs --> General logs and tensorboard logs 
    for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs,local_data_dirs), total=2, desc ="list of folders",colour = "red"):
        #create folder to copy the data
        create_directory([local_data_dir])
        copy_file(source_download_dir,local_data_dir)
        
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    args.add_argument("--config","-c",default="config/config.yaml")
    
    args.add_argument("--params","-p",default="params.yaml")

    
    parsed_args = args.parse_args()
    
    try:
        # print_fn("test") # to check exception
        logging.info("\n>>>>>>>>>>>>>>>>stage two started")
        prepare_base_model(config_path=parsed_args.config, params_path= parsed_args.params)
        logging.info("stage two completed! base model is created>>>>>>>>>>>>>>>>>>>>")
        
    except Exception as e:
        logging.exception(e)
        raise e   