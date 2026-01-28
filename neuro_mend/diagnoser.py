import torch
import umap
import hdbscan
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm

class Diagnoser:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"[*] Initializing Diagnoser with DINOv2-Small on {device}...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
        self.model.eval()

    def get_embeddings(self, image_paths, batch_size=32):
        embeddings = []
        print(f"[*] Extracting embeddings for {len(image_paths)} images...")
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the CLS token (index 0) as the image fingerprint
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(emb)
            
        return np.vstack(embeddings)

    def cluster_failures(self, embeddings, min_cluster_size=10):
        print("[*] Running UMAP reduction...")
        # Reduce to 5 dims first for better clustering density
        reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
        reduced_emb = reducer.fit_transform(embeddings)

        print("[*] Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(reduced_emb)
        
        # Organize results
        unique_labels = set(labels)
        results = {}
        for label in unique_labels:
            if label == -1: continue # Noise
            indices = np.where(labels == label)[0]
            results[f"cluster_{label}"] = indices.tolist()
            
        print(f"[+] Found {len(results)} failure clusters.")
        return results
