from glob import glob
from math import floor

import cv2
import numpy as np
import torch
from scipy.optimize import differential_evolution
from torchvision.transforms import ToTensor, ToPILImage, Resize
from tqdm import tqdm

from interactive_canny import get_imgs
from net.model import MyModel
from net.tuple_dataset import CropZeros
from utils.render import json_to_verts_faces, construct_mesh, get_renderer
from utils.model_subset import get_models_subset
from utils.utils import get_kpts, reproj_err


class Transform:
    def __init__(self):
        self.transforms = [CropZeros(),
                           ToPILImage(),
                           Resize((512, 512)),
                           ToTensor()]

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def error(x, gt_img, renderer, meshes, model, transform):
    dist = 5
    class_id = 10
    elev, azim = x
    fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi
    img_pred = get_imgs(meshes[floor(class_id)], renderer, dist, elev, azim, fov)[1]

    diff = model(transform(gt_img).unsqueeze(0).cuda(), transform(img_pred).unsqueeze(0).cuda())

    gt_img1 = transform(gt_img)
    img_pred1 = transform(img_pred)
    # visualize
    cv2.imshow("gt_img", gt_img1.numpy().squeeze())
    cv2.imshow("img_pred", img_pred1.numpy().squeeze())
    cv2.waitKey(1)
    return diff.mean().item()


def main():
    n_classes = 20
    subset = get_models_subset(n_classes)
    meshes = [construct_mesh(*json_to_verts_faces(path)) for i, path in enumerate(glob("data/objs/*.json")) if
              i in subset]
    renderer = get_renderer(imsize=512)
    dist, elev, azim, class_id = 40, 15, 45, 10
    fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi
    gt_img = get_imgs(meshes[class_id], renderer, dist, elev, azim, fov)[1]

    model = MyModel(out_channels=2, multihead=True).cuda()
    model.load_state_dict(torch.load("model_ae20k.pth"))
    model.eval()

    transform = Transform()

    bounds = [(3, 45), (0, 360)]
    # bounds = [(10, 20), (15, 90)]
    # results = dual_annealing(reproj_err, bounds=bounds, args=(gt_points, renderer, meshes), maxiter=200)
    # differential evolution
    # start = time.time()
    results = differential_evolution(error, bounds=bounds, args=(gt_img, renderer, meshes, model, transform),
                                     popsize=8, workers=1, maxiter=50, polish=False, disp=True)  # , atol=0.1
    # print(time.time() - start)

    np.set_printoptions(precision=3)
    print(repr(results.x))


if __name__ == '__main__':
    main()
