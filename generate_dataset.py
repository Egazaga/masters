import os
import random
from glob import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    # dist_range, elev_range, azim_range, class_id_range = (4.5, 12), (15, 15), (0, 360), (0, len(meshes) - 1)
    # dist_range, elev_range, azim_range, class_id_range = (7, 7), (15, 15), (0, 360), (0, len(meshes) - 1)
    # dist_range, elev_range, azim_range, class_id_range = (7, 7), (3, 45), (45, 45), (0, len(meshes) - 1)
    # dist_range, elev_range, azim_range, class_id_range = (4.5, 12), (15, 15), (45, 45), (0, len(meshes) - 1)
    dist_range, elev_range, azim_range, class_id_range = (7, 7), (3, 45), (0, 360), (0, len(meshes) - 1)
    output_dir = "data/eac20k/"
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print("Output directory already exists. Exiting.")
        exit(1)

    values = []
    for i in tqdm(range(20000)):
        label1, label2 = [], []
        for a, b in [dist_range, elev_range, azim_range]:
            diff = np.random.uniform(0, b - a)
            val1 = np.random.uniform(a, b - diff)
            val2 = val1 + diff
            if random.random() > 0.5:  # to balance l1 and l2 distributions
                val1, val2 = val2, val1

            label1.append(val1)
            label2.append(val2)

        class_id = np.random.randint(*class_id_range)
        label1.append(class_id)
        if random.random() > 0.5:
            label2.append(class_id)
        else:
            label2.append(np.random.randint(*class_id_range))

        dist, elev, azim, class_id = label1
        fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi
        img1 = get_imgs(meshes[class_id], renderer, dist, elev, azim, fov)[1]

        dist2, elev2, azim2, class_id = label2
        fov2 = 2 * np.arctan(0.5 * 6 / dist2) * 180 / np.pi
        img2 = get_imgs(meshes[class_id], renderer, dist2, elev2, azim2, fov2)[1]

        # create folder
        os.makedirs(output_dir + "/pic/", exist_ok=True)
        out_img = np.stack((img1, img2, np.zeros_like(img1)), axis=2)
        cv2.imwrite(output_dir + "/pic/" + str(i).zfill(5) + ".png", out_img)
        values.append([label1, label2])

    # save values
    np.save(output_dir + "/data.npy", np.array(values))

    # plt.hist(np.abs(np.array(values)[:, 0, 0] - np.array(values)[:, 1, 0]), bins=50)
    # plt.hist(np.array(values)[:, 0, 0], bins=50)
    # plt.show()
