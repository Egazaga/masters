import json
import os
from glob import glob

import numpy as np
from sklearn_extra.cluster import KMedoids


def read_kpts_from_our_json(path):
    with open(path) as json_file:
        data = json.load(json_file)

    keypoints_dict = data['pts'] if 'pts' in data else None
    if keypoints_dict is None:
        print("No kpts in", path)
    if None not in keypoints_dict.values():
        all_pts = data['vertices']
        keypoints = [all_pts[i] for i in keypoints_dict.values() if isinstance(i, int)]
    else:
        print("not 32 points in", path, keypoints_dict)

    keypoints = np.array(keypoints)
    keypoints[:, 1] *= -1  # models are "standing on the roof"

    return keypoints


def _get_models_subset(cached_models, n_models):
    models_size = len(cached_models)
    distance_matrix = np.zeros((models_size, models_size), dtype=float)
    for i in range(0, models_size):
        for j in range(i + 1, models_size):
            distance_matrix[i, j] = np.sqrt(
                np.sum(np.square(cached_models[i] - cached_models[j]), axis=1)).sum()

    distance_matrix += distance_matrix.T

    km_model = KMedoids(n_clusters=n_models, random_state=0, metric='precomputed', method='pam',
                        init='k-medoids++').fit(distance_matrix)

    # print(km_model.medoid_indices_)
    # print(np.array(os.listdir("models"))[km_model.labels_ == 9])
    # (unique, counts) = np.unique(km_model.labels_, return_counts=True)
    # frequencies = np.asarray((unique, counts)).T
    # print("Counts:\n", frequencies)
    return km_model.medoid_indices_


def get_models_subset(n_models):
    cached_models = [read_kpts_from_our_json(path) for path in glob("data/models/*.json")]
    return _get_models_subset(cached_models, n_models)

