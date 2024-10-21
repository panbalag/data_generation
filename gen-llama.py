from transformers import AutoTokenizer
from transformers import pipeline
import json
import torch
import re

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Choose a model. For faster runtime use 7b model. Note that quality
# of the generated content may be impacted
# Models:
#   meta-llama/Llama-2-13b-chat-hf
#   meta-llama/Llama-2-7b-chat-hf

model = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, token=True)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt):

    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=4096,
    )
    return(sequences[0]['generated_text'])

# Define the prompt to generate content. Here are some different prompts used to generate
# contents w.r.to walkthrough guides, cheatsheets, hardware grading, maintenance, etc.,
# Replace the prompt to generate the appropriate content
# Prompts:
#  1. walkthrough guide: "'Write a detailed game walkthrough guide for the game '+ name
#  2. blogs (games): 'Write a detailed blog on the game '+ name + ' focusing on gameplay, story line, style, sound and music, innovation, etc'
#  3. blogs (gaming system): 'Write a detailed blog on the fictional gaming system '+ name + ' focusing on hardware specifications, unique features, notable games, etc'
#  4. HW buying guide: 'Write a comprehensive buying guide for the gaming system '+ name
#  5. HW condition and grading guide: 'Write a comprehensive hardware condition and grading guide for the gaming system '+ name
#  6. HW controller repair and maintenance guide: 'Write a comprehensive controller repair and maintenance guide for the gaming system '+ name + 'focusing on common issues, repair, cleaning and maintenance'
#  7. HW restoration guide: 'Write a comprehensive hardware restoration guide for the fictional gaming system '+ name + 'focusing on patching and updates, hardware, software, sound and music restoration'
#  8. 'Write a comprehensive troubleshooting guide for the retro gaming system '+ name + 'focusing on power issues, display problems, audio issues, game crashes, overheating, networking issues, controller problems, performance issues and how to troubleshoot and fix them'

def gen_content(name):
  prompt = 'Write a detailed game walkthrough guide for the game '+ name
  print("Generating game walkthrough guide for "+ name)
  content = get_llama_response(prompt)
  file_name = re.sub(r'[^A-Za-z0-9]', '', name)
  file_name = file_name[:20] if len(file_name) > 20 else file_name

  # File path to save generated content.
  # Important: Change the path depending on content generated.

  file_path = "content/tmp/" + file_name +".txt"  # You can change this to your desired file name or path
  print(f"Writing to {file_path}")
  # Save the content to file
  with open(file_path, 'w') as file:
      file.write(str(content))
  print(f"Content written to {file_path}")


file_path = 'names.json'
keys_to_find = ['games_retro', 'games_fictional']
# keys_to_find = ['gaming_systems_retro', 'gaming_system_fictional']
data = read_json_file(file_path)
for item in data['games_retro']:
  gen_content(item)
for item in data['games_fictional']:
  gen_content(item)

#for item in data['gaming_systems_retro']:
#  gen_content(item)
#for item in data['games_fictional']:
#  gen_content(item)
