import torch
import tiktoken
import math
from model import TransformerLanguageModel  # Make sure to import your model correctly

# Hyperparameters (should match the training script)
d_model = 64
context_length = 16
num_heads = 4
max_token_value = 50257  # Adjust according to your training data (if different)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer and model
encoding = tiktoken.get_encoding("cl100k_base")
model = TransformerLanguageModel()
model.load_state_dict(torch.load('model-ckpt.pt', map_location=device))
model.to(device)

# Function for generating text
def generate_text(start_text, max_new_tokens=100):
    start_ids = encoding.encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # Add batch dimension
    y = model.generate(x, max_new_tokens=max_new_tokens)
    generated_text = encoding.decode(y[0].tolist())  # Decode generated tokens
    return generated_text

# Test the model with an example input
if __name__ == "__main__":
    start_text = "The salesperson"
    print("Generating text from the model...")
    generated_text = generate_text(start_text, max_new_tokens=100)
    print("Generated Text:")
    print(generated_text)
