from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import re

#model_name = "meta-llama/Llama-2-13b-hf"
model_name = "meta-llama/Llama-2-7b-hf"
#model_name="mistralai/Mistral-7B-v0.1"
#model_name = "fbellame/llama2-pdf-to-quizz-13b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

def write_generated_content_to_single_file(generated_content, output_file="output.txt"):
    # Write the generated content to the single output file (append mode)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_content + "\n\n\n\n")  # Add some spacing between entries
    
    print(f"Appended generated Q&A to: {output_file}")

def generate_qa_pairs(text, model, tokenizer, max_length=1024):
    new_text = " ".join(text.split())

    prompt = f"""Generate a question and answer for the context provided under Now similar to Example1 and Example2.
Example1:
Context: The Amazon rainforest is the largest tropical rainforest in the world, spanning across several countries in South America. It is known for its biodiversity and plays a crucial role in regulating the Earth's oxygen and carbon cycles.
Question: What is the Amazon rainforest known for?
Answer: The Amazon rainforest is known for its biodiversity and its role in regulating Earth's oxygen and carbon cycles.

Example2:
Context: Photosynthesis is a crucial biological process that allows plants, algae, and certain bacteria to convert light energy into chemical energy. It is the foundation of life on Earth, as it is responsible for producing the oxygen we breathe and the food we consume. The process primarily takes place in the chloroplasts of plant cells, specifically within the green pigment chlorophyll, which captures sunlight. Photosynthesis occurs in two main stages: the light-dependent reactions and the light-independent reactions, or the Calvin cycle. In the light-dependent reactions, which occur in the thylakoid membranes of the chloroplast, sunlight is absorbed by chlorophyll and other pigments. This energy is used to split water molecules (H₂O) into oxygen, protons, and electrons. The oxygen is released as a byproduct, while the energy from the electrons is used to create energy-rich molecules, such as ATP (adenosine triphosphate) and NADPH (nicotinamide adenine dinucleotide phosphate). The Calvin cycle, which takes place in the stroma of the chloroplast, uses the ATP and NADPH generated from the light-dependent reactions to convert carbon dioxide (CO₂) into glucose, a type of sugar. During this stage, carbon dioxide from the atmosphere is captured and, through a series of enzyme-driven reactions, is incorporated into organic molecules. These molecules are eventually converted into glucose, which plants use as an energy source for growth, reproduction, and other functions. Photosynthesis is vital not only for plants but for nearly all life on Earth. By producing oxygen and organic compounds, it supports ecosystems and maintains the balance of gases in the atmosphere.Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities. This process occurs mainly in the chloroplasts of plant cells. Without photosynthesis, the Earth's food chain and oxygen levels would collapse.
Photosynthesis is a fundamental process that sustains life on Earth by converting sunlight into chemical energy. It occurs primarily in plants, algae, and certain bacteria, where chlorophyll captures sunlight to produce glucose (a sugar) and oxygen from carbon dioxide and water. This process is vital for several reasons. Firstly, photosynthesis is the basis of the food chain. The glucose produced by plants serves as an energy source for them, allowing growth and development. Herbivores, in turn, consume plants, gaining the energy they need, which then passes on to carnivores and omnivores. Without photosynthesis, life forms that depend on plant-based energy would not survive, leading to the collapse of ecosystems. Secondly, photosynthesis plays a critical role in regulating Earth's atmosphere. The oxygen released as a byproduct of photosynthesis is essential for most living organisms to breathe. Before photosynthesis evolved, the Earth's atmosphere had very little oxygen. Today, it is responsible for maintaining the oxygen levels that sustain aerobic life, including humans. In addition, photosynthesis helps remove carbon dioxide, a greenhouse gas, from the atmosphere. As plants absorb CO2 during the process, they mitigate the effects of climate change. Forests and oceans, which are rich in photosynthetic organisms, act as major carbon sinks, helping to balance Earth's carbon cycle. Finally, photosynthesis is crucial to agriculture, the foundation of human civilization. Crops rely on this process to grow and produce food. Advances in agricultural science, such as understanding how to optimize photosynthesis, can increase food production to meet the needs of the growing global population.
Question: Where does photosynthesis occur in plant cells?
Answer: Photosynthesis occurs mainly in the chloroplasts of plant cells.

Now:
Generate a question and answer based for the context below
Context: {new_text}
"""


    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
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

