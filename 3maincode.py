import re
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Dataset of mathematical problem statements and operations
# Replace this with your own dataset
math_dataset = [
    ("What is the result of adding 2 and 3?", "add"),
    ("Subtract 5 from 10.", "subtract"),
    ("Multiply 4 by 7.", "multiply"),
    ("Divide 15 by 3.", "divide"),
    # Add more examples as needed
]

# Tokenize the dataset
tokenized_dataset = tokenizer([item[0] for item in math_dataset], truncation=True, padding=True)

# Define a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_dataset["input_ids"][idx],
            "attention_mask": self.tokenized_dataset["attention_mask"][idx],
        }

# Prepare the training data
train_data = CustomDataset(tokenized_dataset)

# Fine-tune the GPT-2 model on the dataset
training_args = TrainingArguments(
    output_dir='./results',  # Provide the output directory
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# Function to preprocess input text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert text to lowercase
    return text

# Function to generate C++ code based on the input text
def generate_cpp_code(numbers, operation):
    operation_str = ["+", "-", "*", "/"][operation]
    numbers_str = " ".join(str(num) for num in numbers)
    cpp_code = f"#include <iostream>\n\nint main() {{\n\tint result = {numbers_str} {operation_str};\n\tstd::cout << \"Result: \" << result << std::endl;\n\treturn 0;\n}}"
    return cpp_code

# Function to write generated C++ code to a file
def write_cpp_code_to_file(cpp_code, filename="generated_code.cpp"):
    with open(filename, 'w') as file:
        file.write(cpp_code)
    print(f"Generated C++ code written to {filename}")

# Main function to run the entire process
def generate_cpp_code_from_input(input_text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(input_text)
        
        # Extract numbers from the input text
        numbers = [int(num) for num in re.findall(r'\b\d+\b', preprocessed_text)]

        # Detect the operation from the input text
        if "add" in preprocessed_text:
            operation = 0
        elif "subtract" in preprocessed_text:
            operation = 1
        elif "multiply" in preprocessed_text:
            operation = 2
        elif "divide" in preprocessed_text:
            operation = 3
        else:
            print("Invalid input. Please provide a valid mathematical operation.")
            return
        
        if len(numbers) < 2:
            print("Invalid input. Please provide at least two numbers.")
            return

        # Generate C++ code based on the operation
        cpp_code = generate_cpp_code(numbers, operation)

        # Write the generated C++ code to a file
        write_cpp_code_to_file(cpp_code)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_text = input("Please enter a mathematical problem to solve: ")
generate_cpp_code_from_input(input_text)
