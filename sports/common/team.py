from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipVisionModel
import joblib

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        self.features_model.eval()
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt", padding=True
                ).to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)
    
    def save(self, path: str) -> None:
        """
        Persist the reducer and clustering to disk.
        """
        joblib.dump({
            'umap':   self.reducer,
            'kmeans': self.cluster_model
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu', batch_size: int = 32):
        """
        Instantiate a fresh object and restore its reducer + cluster_model.
        """
        obj = cls(device=device, batch_size=batch_size)
        data = joblib.load(path)
        obj.reducer        = data['umap']
        obj.cluster_model  = data['kmeans']
        # No need to re-load the vision model; it's already in __init__
        return obj
