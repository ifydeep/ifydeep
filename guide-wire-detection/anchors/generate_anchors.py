import os, glob, json, tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def get_bboxes_wh(data_path: str, bbox_subfolder_name: str="BBox") -> np.ndarray:
    sample_subfolders = [
        os.path.join(data_path, path) for path in os.listdir(data_path)
    ]
    bboxes = []
    for path in tqdm.tqdm(sample_subfolders):
        bbox_paths = glob.glob(os.path.join(path, bbox_subfolder_name, "*.txt"), recursive=True)

        for bbox_path in bbox_paths:
            with open(bbox_path, "r") as f:
                bbox = tuple(map(lambda x : float(x), f.read().split()[1:]))
            f.close()
            bboxes.append(bbox)

    return np.array(bboxes)[:, 2:]


def compute_anchors(
        bboxes: np.ndarray, 
        k: int, 
        tol: float=1e-6, 
        max_iter: int=500, 
        random_state: int=42) -> Tuple[np.ndarray, float]:
    
    assert k % 3 == 0, f"value of k must be divisible by 3, got {k}"
    
    cluster_model = KMeans(
        n_init="auto", 
        n_clusters=k, 
        max_iter=max_iter, 
        tol=tol, 
        random_state=random_state)
    
    cluster_model.fit(bboxes)
    silhouette = silhouette_score(bboxes, cluster_model.labels_)
    return cluster_model.cluster_centers_, silhouette


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(message)s")
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    bbox_dir = os.path.join(script_dir, "../data/Train_data")
    anchor_box_file = os.path.join(script_dir, "anchors.json")

    if not os.path.isfile(anchor_box_file):
        k = 9
        tol = 1e-7
        max_iter = 500
        random_state = 42

        logging.info(f"Extracting bounding boxes from '{bbox_dir}'")
        bboxes = get_bboxes_wh(bbox_dir)
        scaler = StandardScaler()
        bboxes = scaler.fit_transform(bboxes)

        logging.info(f"Generating anchors with {KMeans.__name__} clustering algorithm")
        centroids, _ = compute_anchors(bboxes, k, tol, max_iter, random_state)
        centroids = scaler.inverse_transform(centroids)
        sorted_anchors = sorted(centroids, key=lambda centroid : centroid.prod())

        sm_anchors = sorted_anchors[:3]     # small scale anchor boxes
        md_anchors = sorted_anchors[3:6]    # small scale anchor boxes
        lg_anchors = sorted_anchors[6:]     # small scale anchor boxes

        logging.info(f"Storing anchors in {anchor_box_file} file")
        anchor_boxes = {
            "small anchors": np.array(sm_anchors).tolist(),
            "mid anchors": np.array(md_anchors).tolist(),
            "large anchors": np.array(lg_anchors).tolist()
        }

        with open(anchor_box_file, "w") as f:
            json.dump(anchor_boxes, f)
        f.close()

    else:
        logging.info(f"Anchors have already been generated")