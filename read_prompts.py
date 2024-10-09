import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to print each element in the JSON object
def print_json_elements(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print_json_elements(value, indent + 1)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print_json_elements(item, indent + 1)
    else:
        print('  ' * indent + str(data))

file_path = 'prompts.json'
try:
    data = read_json_file(file_path)
    # Call the function to print the JSON elements
    print_json_elements(data)
except FileNotFoundError:
    print(f"The file at {file_path} was not found.")
except json.JSONDecodeError:
    print(f"Error decoding JSON from the file at {file_path}.")

