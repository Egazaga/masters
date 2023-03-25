import os
from glob import glob

import cv2
import numpy as np

from interactive_canny import get_imgs
from utils.model_subset import get_models_subset
from utils.render import json_to_verts_faces, construct_mesh, get_renderer

if __name__ == '__main__':
    subset = get_models_subset(20)
    meshes = [construct_mesh(*json_to_verts_faces(path)) for i, path in enumerate(glob("data/objs/*.json")) if
              i in subset]
    renderer = get_renderer(imsize=512)
    dist_range, elev_range, azim_range, class_id_range = (4.5, 12), (3, 45), (0, 360), (0, len(meshes) - 1)

    values = []
    for i in range(1000):
        dist, elev, azim, class_id = np.random.uniform(*dist_range), np.random.uniform(*elev_range), \
            np.random.uniform(*azim_range), np.random.randint(*class_id_range)

        fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi

        canny = get_imgs(meshes[class_id], renderer, dist, elev, azim, fov)[1]

        # create folder
        os.makedirs(f"data/dataset/canny", exist_ok=True)
        cv2.imwrite(f"data/dataset/canny/{str(i).zfill(5)}.png", canny)
        values.append([dist, elev, azim, class_id, fov])

    # save values
    np.save("data/dataset/canny.npy", np.array(values))
