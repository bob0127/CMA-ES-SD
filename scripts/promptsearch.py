import torch
import numpy as np
import clip

class PromptSearcher:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        # 載入儲存好的 features
        data = np.load("prompt_features.npz")
        keys = list(data.keys())
        features = [data[k] for k in keys]
        features = np.concatenate(features, axis=0)  # 變成 shape: (300, 512)
        
        self.features = features / np.linalg.norm(features, axis=1, keepdims=True)

    def find_most_similar(self, new_prompt):
        with torch.no_grad():
            token = clip.tokenize([new_prompt]).to(self.device)
            new_feature = self.model.encode_text(token)
            new_feature = new_feature / new_feature.norm(dim=-1, keepdim=True)
            new_feature_np = new_feature.cpu().numpy().squeeze()  # shape: (512,)

        # 計算 cosine similarity
        sims = self.features @ new_feature_np  # shape: (300,)，每個都是 cosine similarity

        # 找最大相似度的 index
        best_index = np.argmax(sims)
        return best_index