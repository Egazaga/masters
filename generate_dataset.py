import os
import random
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from interactive_canny import get_imgs
from utils.model_subset import get_models_subset
from utils.render import json_to_verts_faces, construct_mesh, get_renderer

if __name__ == '__main__':
    subset = get_models_subset(20)
    meshes = [construct_mesh(*json_to_verts_faces(path)) for i, path in enumerate(glob("data/objs/*.json")) if
              i in subset]
    renderer = get_renderer(imsize=512)
    # dist_range, elev_range, azim_range, class_id_range = (4.5, 12), (3, 45), (0, 360), (0, len(meshes) - 1)
    dist_range, elev_range, azim_range, class_id_range = (7, 7), (15, 15), (0, 360), (0, len(meshes) - 1)

    values = []
    for i in tqdm(range(10000)):
        dist, elev, azim, class_id = np.random.uniform(*dist_range), np.random.uniform(*elev_range), \
            np.random.uniform(*azim_range), np.random.randint(*class_id_range)

        fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi
        img1 = get_imgs(meshes[class_id], renderer, dist, elev, azim, fov)[1]
        label1 = [dist, elev, azim, class_id]

        diff_scale = 0.5
        new_vals = []
        for orig, val_range in zip((dist, elev, azim), (dist_range, elev_range, azim_range)):
            deviation = (val_range[1] - val_range[0]) * diff_scale * random.random()
            if random.random() > 0.5:
                deviation *= -1
            new_vals.append(orig + deviation)
        dist2, elev2, azim2 = new_vals
        fov2 = 2 * np.arctan(0.5 * 6 / dist2) * 180 / np.pi
        img2 = get_imgs(meshes[class_id], renderer, dist2, elev2, azim2, fov2)[1]
        label2 = [dist2, elev2, azim2, class_id]

        # create folder
        os.makedirs(f"data/dataset/canny", exist_ok=True)
        out_img = np.stack((img1, img2, np.zeros_like(img1)), axis=2)
        cv2.imwrite(f"data/dataset/canny/{str(i).zfill(5)}.png", out_img)
        values.append([label1, label2])

    # save values
    np.save("data/dataset/canny.npy", np.array(values))
