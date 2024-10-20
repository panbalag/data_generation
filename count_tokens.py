import os
from transformers import LlamaTokenizer

# Load the LLaMA2 tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

def count_tokens_in_file(file_path):
    """Reads a file, tokenizes its content, and returns the number of tokens."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Tokenize the text and count the number of tokens
    tokens = tokenizer.encode(text)
    return len(tokens)

def count_tokens_in_directory(directory_path):
    """Recursively counts tokens in all .txt files within a directory and its subdirectories."""
    total_tokens = 0
    # Traverse the directory tree using os.walk
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):  # Only process .txt files
                file_path = os.path.join(root, file)
                tokens_in_file = count_tokens_in_file(file_path)
                total_tokens += tokens_in_file
                print(f"File: {file_path} -> {tokens_in_file} tokens")
    
    return total_tokens

if __name__ == "__main__":
    # Replace with the path to your directory
    directory = "content"  
    total_tokens = count_tokens_in_directory(directory)
    print(f"Total tokens in all .txt files: {total_tokens}")

