This repo contains scripts to automatically generate data using LLAMA2 13 Model.

Requirements:

1) RHEL with GPU
     The code in this repo has been tested with NVIDIA L4 and CUDA 12.4
2) Pip
     python -m ensurepip --upgrade
3) Pytorch (CUDA 12.4)
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     For older versions of CUDA, refer 
     https://pytorch.org/get-started/locally/

Steps:

1) The user will need to install the library to interact with the Hugging Face Hub and login to Hugging Face in order to access the model. 
   pip install huggingface_hub
   huggingface-cli login
2) python gen-llama.py

