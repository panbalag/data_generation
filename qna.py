from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json

#model_name = "meta-llama/Llama-2-13b-hf"
#model_name = "meta-llama/Llama-2-7b-hf"
#model_name="mistralai/Mistral-7B-v0.1"
model_name = "fbellame/llama2-pdf-to-quizz-13b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

def write_generated_content_to_single_file(generated_content, output_file="output.txt"):
    # Write the generated content to the single output file (append mode)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_content + "\n\n\n\n")  # Add some spacing between entries
    
    print(f"Appended generated Q&A to: {output_file}")

def generate_qa_pairs(text, model, tokenizer, max_length=1024):
    prompt = f"You are a teacher preparing questions for a quiz. Given the following document, please generate 1 multiple-choice questions (MCQs) with 4 options and a corresponding answer letter based on the document \n Example question \n Question: question here \n CHOICE_A: choice here \n CHOICE_B: choice here \n CHOICE_C: choice here \n CHOICE_D: choice here \n Answer: A or B or C or D \n  These questions should be detailed and solely based on the information provided in the document.\n <Begin Document> {text} <End Document>"

    #inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7)
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("===============GENERATED TEXT ==================================")
    print(generated_text)
    result = generated_text.replace(prompt, "").strip()
    print("===============RESULT ==================================")
    print(result)
    return result
    
folder_path = 'content/blog/'
output_path = 'qna/blog/'
for root, dir, files in os.walk(folder_path):
    for file in files:
        # Open and read the file
        file_path = folder_path + file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Generating Q&A for {file_path}:")
            qna =  generate_qa_pairs(content, model, tokenizer)          
            output_file = output_path + file
            write_generated_content_to_single_file(qna, output_file)

