from transformers import AutoTokenizer
from transformers import pipeline
import json
import torch
import re

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

model = "meta-llama/Llama-2-13b-chat-hf"
#model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, token=True)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt):
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """

    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=4096,
    )
    return(sequences[0]['generated_text'])

def gen_content(name):
  #prompt = 'Write a comprehensive buying guide for the gaming system '+ name
  prompt = 'Write a comprehensive controller repair and maintenance guide for the gaming system '+ name + 'focusing on common issues, repair, cleaning and maintenance'
  print("Generating buying guide for "+ name)
  content = get_llama_response(prompt)
  file_name = re.sub(r'[^A-Za-z0-9]', '', name)
  file_name = file_name[:20] if len(file_name) > 20 else file_name
  file_path = "content/hw_repair/" + file_name +".txt"  # You can change this to your desired file name or path
  print(f"Writing to {file_path}")
  # Save the content to file
  with open(file_path, 'w') as file:
      file.write(str(content))
  print(f"Content written to {file_path}")


file_path = 'names.json'
keys_to_find = ['gaming_systems_retro', 'gaming_system_fictional']
data = read_json_file(file_path)
for item in data['gaming_systems_retro']:
  gen_content(item)
#for item in data['gaming_systems_fictional']:
#  gen_content(item)
