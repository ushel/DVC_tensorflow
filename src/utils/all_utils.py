import yaml
import os
import json
import logging
import time

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directory(dirs:list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        # print(f"Directory created at {dir_path}")
        logging.info(f"Directory created at {dir_path}")
        
def save_local_df(data,data_path,index_status=False):
    data.to_csv(data_path, index=index_status)
    # print(f"Data is saved at {data_path}")
    logging.info(f"Data is saved at {data_path}")
    
def save_reports(report:dict, report_path:str,indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    # print(f"reports are saved at {report_path}")
    logging.info(f"reports are saved at {report_path}")
    
def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name