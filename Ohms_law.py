import re
import math
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Initialize the language model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

def extract_values_from_text(text):
    #Extract V, R and I values
    V = re.search(r'V\s*=\s*([\d.]+)\s*V', text, re.IGNORECASE)
    R = re.search(r'R\s*=\s*([\d.]+)\s*(kΩ|Ω)', text, re.IGNORECASE)
    I = re.search(r'I\s*=\s*([\d.]+)\s*(mA|A)', text, re.IGNORECASE)
    print(V,R,I)

    if V and R:
        V = float(V.group(1))
        if R.group(2).endswith('kΩ'):
            R = float(R.group(1)) * 1000  #Convert to Ω
        else:
            R = float(R.group(1))
        return V,R,None
    
    if V and I:
        V = float(V.group(1))
        if I.group(2).endswith('mA'):
            I = float(I.group(1)) * 1e-3
        else:
            I = float(I.group(1))
        return V, None, I
    
    if R and I:
        if R.group(2).endswith('kΩ'):
            R = float(R.group(1)) * 1000  #Convert to Ω
        else:
            R = float(R.group(1))
        if I.group(2).endswith('mA'):
            I = float(I.group(1)) * 1e-3
        else:
            I = float(I.group(1))
        return None, R, I
    


    
def R_calculator(V,I):
    return V/I 

def V_calculator(R,I):
    return I*R

def I_calculator(V,R):
    return V/R

def main():
    #User input
    user = input("Enter you question: ")
    # Use the language model to parse and understand the input
    response = nlp(user, max_length=100, num_return_sequences=1)
    parsed_text = response[0]['generated_text']

    print(parsed_text)
    #Extracting values
    V,R,I = extract_values_from_text(parsed_text)


    if V is None:
        Voltage = V_calculator(R,I)
        print(f"The Voltage is {Voltage} V.")
    
    elif R is None:
        Resistance = R_calculator(V,I)
        print(f"The Resistance is {Resistance} Ω.")

    elif I is None:
        Current = I_calculator(V,R)
        print(f"The current is {Current} A.")

    else:
        print("Not able to parse the question for potential difference, resistance or current!")

if __name__ == "__main__":
    main()