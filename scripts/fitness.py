import cv2
import numpy as np
from PIL import Image
from scipy.special import expit as sigmoid
import torch
import clip

class FitnessEvaluator:
    def __init__(self, model, preprocess, device, alpha=0.7, beta=0.1, gamma = 0.2):
        self.device = device
        self.model =  model
        self.preprocess = preprocess
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def get_blur_score(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.variance_of_laplacian(gray)

    def get_clip_score(self, image_path, prompt):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            similarity = torch.cosine_similarity(image_features, text_features).item()
        return similarity
    
    def get_image_score(self, image_path1, image_path2):
        image1 = self.preprocess(Image.open(image_path1)).unsqueeze(0).to(self.device)
        feature = np.load("image_features.npz")
        with torch.no_grad():
            image_feature1 = self.model.encode_image(image1)
            image_feature2 = torch.tensor(feature[image_path2]).to(self.device)
            similarity = torch.cosine_similarity(image_feature1, image_feature2).item()
        return similarity

    def compute(self, image_path1, image_path2, prompt):
        clip_score = self.get_clip_score(image_path1, prompt)
        blur_score = self.get_blur_score(image_path1)
        image_score = self.get_image_score(image_path1, image_path2)
        blur_normalized = sigmoid(np.log(blur_score + 1))
        fitness = self.alpha * clip_score + self.beta * blur_normalized + self.gamma * image_score
        return -fitness
