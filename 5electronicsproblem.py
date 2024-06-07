import re
import math
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Initialize the language model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

def extract_values_from_text(text):
    # Extract resistor and capacitor values using regex
    R_match = re.search(r'R\s*=\s*([\d.]+)\s*kΩ', text, re.IGNORECASE)
    C_match = re.search(r'C\s*=\s*([\d.]+)\s*µF', text, re.IGNORECASE)

    if R_match and C_match:
        R_value = float(R_match.group(1)) * 1e3  # convert kΩ to Ω
        C_value = float(C_match.group(1)) * 1e-6  # convert µF to F
        return R_value, C_value
    else:
        return None, None

def calculate_cutoff_frequency(R, C):
    return 1 / (2 * math.pi * R * C)

def main():
    # User input
    user_input = input("Please enter your electronics problem: ")

    # Use the language model to parse and understand the input
    response = nlp(user_input, max_length=100, num_return_sequences=1)
    parsed_text = response[0]['generated_text']

    # Extract values
    R, C = extract_values_from_text(parsed_text)

    if R is not None and C is not None:
        # Calculate the cutoff frequency
        f_c = calculate_cutoff_frequency(R, C)
        print(f"The cutoff frequency of the RC filter is approximately {f_c:.2f} Hz.")
    else:
        print("Could not extract resistor and capacitor values from the input.")

if __name__ == "__main__":
    main()
