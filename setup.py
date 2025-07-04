from setuptools import setup

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()
    
    
setup(
    name = "src",
    version="0.0.1",
    author="utkarsh",
    description="A small package for dvc dl pipeline demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ushel/DVC_tensorflow",
    author_email="utkarsh03@icloud.com",
    packages=["src"],
    license="GNU",
    python_requires=">=3.7",
    install_requires=[
        'dvc'
        'tensorflow'
        'matplotlib'
        'numpy'
        'pandas'
        'tqdm'
        'pyyaml'
        'boto3' 
    ]
)