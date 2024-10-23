import os
import json

# Directory where your files are located
directory_path = 'qna/hw_troubleshoot'

# Output JSON file path
output_file = 'qna_hw_troubleshoot.json'

# List to store all question-answer pairs
qa_pairs = []

# Function to extract question-answer pairs from a file
def extract_qa_pairs(file_content):
    pairs = []
    lines = file_content.splitlines()  # Split file content into lines
    question, answer = None, None
    
    for line in lines:
        if line.startswith("Question:"):
            question = line[len("Question:"):].strip()  # Extract question
        elif line.startswith("Answer:"):
            answer = line[len("Answer:"):].strip()  # Extract answer
            
            # If both question and answer are found, store them as a pair
            if question and answer:
                pairs.append({"question": question, "answer": answer})
                question, answer = None, None  # Reset for the next pair
    return pairs

# Loop over all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Check if the file is not empty and is a file
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            content = file.read()
            pairs = extract_qa_pairs(content)
            qa_pairs.extend(pairs)

# Write all extracted question-answer pairs to a JSON file
with open(output_file, 'w') as json_file:
    json.dump(qa_pairs, json_file, indent=4)

print(f"Extracted {len(qa_pairs)} question-answer pairs to {output_file}")

