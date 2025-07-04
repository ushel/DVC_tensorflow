from src.utils.all_utils import read_yaml, create_directory
import argparse 
import pandas as pd  
import os   
import shutil
from tqdm import tqdm  # print progressbar

def copy_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    for file in tqdm(list_of_files,total=N, desc="copying file from {source_download_dir} to {local_data_dir}", color="green"):
        src = os.path.join(source_download_dir)
        dest = os.path.join(local_data_dir,file)
        shutil.copy(src,dest)
    
    
def get_data(config_path):
    config = read_yaml(config_path)
    
    source_download_dirs = config["source_download_dirs"]
    local_data_dirs = config["local_data_dirs"]
    
    for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs,local_data_dirs), total=2, desc ="list of folders",color = "red"):
        #create folder to copy the data
        create_directory([local_data_dir])
        copy_file(source_download_dir,local_data_dir)