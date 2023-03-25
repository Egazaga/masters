from glob import glob

import numpy as np
from scipy.optimize import differential_evolution
from tqdm import tqdm
from utils.render import json_to_verts_faces, construct_mesh, get_renderer
from utils.model_subset import get_models_subset
from utils import get_kpts, reproj_err


def main():
    n_classes = 20
    n_orb_features = 250
    subset = get_models_subset(n_classes)
    meshes = [construct_mesh(*json_to_verts_faces(path)) for i, path in enumerate(glob("data/objs/*.json")) if i in subset]
    renderer = get_renderer(dist=5, elev=30, azim=45, imsize=256)

    class_preds = []
    for gt_class in tqdm(range(len(meshes) - 1)):
        gt_points = get_kpts(renderer, meshes[gt_class], n_orb_features)

        bounds = [(0, 15), (0, 90), (0, 360), [0, n_classes - 1]]
        # results = dual_annealing(reproj_err, bounds=bounds, args=(gt_points, renderer, meshes), maxiter=200)
        # differential evolution
        # start = time.time()
        results = differential_evolution(reproj_err, bounds=bounds, args=(gt_points, renderer, meshes, n_orb_features),
                                         popsize=16, atol=0.1, workers=1, maxiter=50, polish=False)  # , disp=True
        # print(time.time() - start)

        np.set_printoptions(precision=3)
        print(repr(results.x))
        class_pred = round(results.x[3])
        class_preds.append(class_pred)
        are_equal = [gt == pred for gt, pred in zip(list(range(len(meshes) - 1)), class_preds)]
        print(gt_class, class_pred, sum(are_equal) / len(are_equal))


if __name__ == '__main__':
    main()
