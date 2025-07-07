        
from src.utils.all_utils import read_yaml, create_directory
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
    
def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    
    args.add_argument("--config","-c",default="config/config.yaml")
    
    args.add_argument("--params","-p",default="params.yaml")

    
    parsed_args = args.parse_args()
    
    try:
        # print_fn("test") # to check exception
        logging.info("\n>>>>>>>>>>>>>>>>stage Four started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage Four completed! training completed and model is saved >>>>>>>>>>>>>>>>>>>>")
        
    except Exception as e:
        logging.exception(e)
        raise e    