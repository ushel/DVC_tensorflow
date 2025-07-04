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