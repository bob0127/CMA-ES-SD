import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

def get_prompt_closest(prompts, model, device):
    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = cosine_similarity(text_features.cpu().numpy())

    #closest = []
    #for i in range(len(prompts)):
        #sim_row = similarity[i].copy()
        #sim_row[i] = -1 
        #max_val = sim_row.max()
        #max_idx = sim_row.argmax()
        #closest.append((max_idx,max_val))

    return similarity