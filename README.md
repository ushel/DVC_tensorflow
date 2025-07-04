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