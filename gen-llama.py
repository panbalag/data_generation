from transformers import AutoTokenizer
from transformers import pipeline
import torch

#model = "meta-llama/Llama-2-13b-chat-hf"
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, token=True)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt: str) -> None:
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
    print("Chatbot:", sequences[0]['generated_text'])

prompt = 'Write a game walkthrough guide for a fictional game called Flipping Waffles?\n'
get_llama_response(prompt)

