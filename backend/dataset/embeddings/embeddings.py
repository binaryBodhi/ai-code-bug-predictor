#THIS SCRIPT WAS RUN IN GOOGLE COLAB
#CHECK FILE AI_CODE_REVIEWER_EMBEDDING.ipynb

# import json
# from transformers import RobertaTokenizer, RobertaModel
# import torch
# from tqdm import tqdm
# from app import model, tokenizer

# # Load CodeBERT
# # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# # model = RobertaModel.from_pretrained("microsoft/codebert-base")
# model.eval()

# # File path to CodeXGLUE defect detection dataset
# data_path = "train.jsonl"

# # Output: list of [embedding], and list of labels
# embeddings = []
# labels = []

# with open(data_path, 'r') as f:
#     for line in tqdm(f, desc="Processing code"):
#         item = json.loads(line)
#         code = item["func"]
#         label = item["target"]

#         # Tokenize & truncate
#         inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)

#         with torch.no_grad():
#             outputs = model(**inputs)

#         # Mean pooling over tokens → shape: [1, 768] → squeeze to [768]
#         code_embedding = outputs.last_hidden_state.mean(1).squeeze().numpy()

#         embeddings.append(code_embedding)
#         labels.append(label)
