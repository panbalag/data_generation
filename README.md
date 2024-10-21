This repo contains scripts to automatically generate data using LLAMA2 13B Model. For faster generation, you can switch to LLAMA2 7B.

Layout:

The repo contains generated content (~440K tokens) for real and fictional retro games and gaming systems. The list of games and gaming
systems can be found in names.json.

# Content (content/)
==================

## Game Strategy Consulting: Tips and tricks to master those challenging old-school games
- Game walkthrough guide: `/game_guide`
- Character guide: `/character_guide`
- Cheat sheets: `/cheat_sheet`

## Hardware Sourcing Assistance: Guidance on finding rare and vintage gaming systems
- Buying guide: `/hw_guide`
- Hardware condition and grading guide: `/hw_grading`

## Restoration Support: Advice on restoring and maintaining retro hardware
- Restoration guide: `/hw_restoration`
- Controller repair and maintenance guide: `/hw_repair`
- Troubleshooting guide: `/hw_troubleshoot`

## Community Forums: Connect with fellow retro gamers to share experiences and knowledge
- Blogs on games and gaming systems: `/blog`

Requirements:
=============
1) RHEL with GPU
     The code in this repo has been tested with NVIDIA L4 and CUDA 12.4
2) Pip
     python -m ensurepip --upgrade
3) Pytorch (CUDA 12.4)
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     For older versions of CUDA, refer 
     https://pytorch.org/get-started/locally/
4) git clone https://github.com/panbalag/data_generation.git
4) pip3 install -r requirements.txt
5) huggingface-cli login

Steps:
======
1) The user will need to install the library to interact with the Hugging Face Hub and login to Hugging Face in order to access the model. 
   pip install huggingface_hub
   huggingface-cli login
2) python gen-llama.py

