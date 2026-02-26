import torch
from transformers import AutoTokenizer, AutoModel

class CodeEmbeddingEngine:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.is_loaded = True
        except Exception as e:
            print(f"Warning: Could not load CodeBERT model {e}. Using a dummy embedder for demonstration.")
            self.is_loaded = False

    def get_embedding(self, code_snippet):
        if not self.is_loaded:
            # Fallback to random embeddings if model failed to download/load
            return torch.rand(1, 768).to(self.device)
            
        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token representation as the aggregate embedding
        return outputs.last_hidden_state[:, 0, :]
