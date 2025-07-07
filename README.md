# DVC_tensorflow

instead of creating conda environment in C:// we can create it in project file path...


conda create --prefix ./env python=3.7 -y

from same folder conda activate ./env -if this command fails due to new installation then source activate ./env (other comand)

from other folder conda activate (full path of env) 

this env will not have name in conda env list

if you getting error fsspec

pip uninstall dvc fsspec -y

pip install dvc==2.10.2 fsspec==2022.8.2

alternatively

pip install "dvc[all]"

mkdir -p config src/utils   as we are using src/utils hence use parameter -p


DVC -> DL -> TF 

commands:- 
''' bash

1. Create a new environment

conda create --prefix ./env python=3.7 -y

2. Activate the environment

source activate ./env or conda activate ./env

3. initialize git and dvc

git init

dvc init

4. Create some empty files -

touch 
touch setup.py dvc.yaml params.yaml
mkdir -p config src/utils
touch config/config.yaml
touch config/secrets.yaml

5. Commit on git repository 

git add .

git commit -m "initial setup for project done"

git push origin main

6. Create config file

7. download data -> [source](https://drive.google.com/drive/folders/1tz4IOoJKdi999IRdqJY04VOifyllRzj1)

8 pip install -e . make src as package by installing setup file

if you forget to use f string will print {path_to_yaml}

run above steps 

project --->
# 1. Get data
# 2. Use transfer learning
# 3. Prepare the callbacks tensorboard logs or checkpointing
# 4. Training module(take data from get data section, prepare custom model and take all the call back ---> trained model 
# (file required  - config.yaml, secrets.yaml, params.yaml, artifacts directory,dvc.yaml,loogs.yaml, setup.py))
# artifacts folder --> model, callbacks binary 
# logs --> General logs and tensorboard logs 



Callback functions 

For every epochs will try to store some extra information such as :- 
    training logs - > loss, validation loss, accuracy, validation accuracy
    computation graph of the model you have build.
    its actually a tensorboard callback.

    checkpointing callback -> snapshot of updated model. in between values of weights getting train. during cash will have past history of the model or restart the training.
