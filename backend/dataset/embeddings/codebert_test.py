from transformers import RobertaTokenizer, RobertaModel
import torch

# Load CodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# Example code snippet
code = "def add(a, b): return a + b"

# Tokenize
inputs = tokenizer(code, return_tensors="pt")
outputs = model(**inputs)

# Extract embedding (not human-readable, but can be used for downstream tasks)
print("CodeBERT embedding shape:", outputs.last_hidden_state.shape)