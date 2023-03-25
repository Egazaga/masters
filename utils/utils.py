from collections import defaultdict

import numpy as np
from pytorch3d.renderer import PointLights, look_at_view_transform, FoVPerspectiveCameras
import cv2

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def filter_points(points, tolerance=1):
    point_counts = defaultdict(int)
    min_count = len(points) / 2

    for i, set1 in enumerate(points):
        for point in set1:
            count = 0
            for j, set2 in enumerate(points):
                if i != j and any(
                        abs(point[0] - p[0]) <= tolerance and abs(point[1] - p[1]) <= tolerance for p in set2):
                    count += 1
            if count >= min_count:
                point_counts[point] += 1

    return [point for point, count in point_counts.items() if count >= min_count]


def get_kpts(renderer=None, mesh=None, imgs=None, n_orb_features=250, camera=None):
    if imgs is None:
        d = 3.0
        light_coords = [[0.0, 0.0, -d], [0.0, 0.0, d], [0.0, -d, 0.0], [0.0, d, 0.0], [-d, 0.0, 0.0], [d, 0.0, 0.0]]
        lights = PointLights(device="cuda:0", location=light_coords)
        meshes_ext = mesh.extend(len(light_coords))
        if camera is not None:
            imgs = renderer(meshes_ext, cameras=camera, lights=lights)
        else:
            imgs = renderer(meshes_ext, lights=lights)
        imgs = (imgs[..., :3].cpu().numpy() * 255).astype("uint8")

    orb = cv2.ORB_create(nfeatures=n_orb_features)
    pts = []
    for i, img in enumerate(imgs):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray_image, None)
        pts.append({(round(k.pt[0]), round(k.pt[1])) for k in kp})
    if len(pts) > 1:
        filtered_points = filter_points(pts)
    else:
        filtered_points = pts[0]
    # for pt in filtered_points:
    #     cv2.circle(imgs[0], pt, 3, (0, 255, 0), -1)
    # cv2.imshow('ORB', imgs[0])
    # cv2.waitKey(1)
    return filtered_points


def reproj_err(x_, gt_pts, renderer, meshes, n_orb_features):
    dist, elev, azim, class_id = x_

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    camera = FoVPerspectiveCameras(device="cuda:0", R=R, T=T)

    pts = get_kpts(renderer, meshes[round(class_id)], n_orb_features, camera=camera)
    pts = np.array(pts)
    if len(pts.shape) != 2:
        pts = np.array([[0, 0]])
    loss = pt_distance(gt_pts, pts)

    return loss


def pt_distance(pts1, pts2):
    dist_matrix = cdist(pts1, pts2)

    # score = wasserstein_distance(pts1, pts2)
    idxs = linear_sum_assignment(dist_matrix)
    score = dist_matrix[idxs].sum() / min(len(pts1), len(pts2))
    # penalty for unmatched points
    score += abs(len(pts1) - len(pts2)) ** 2 * 0.05

    return score

# if __name__ == '__main__':
#     verts, faces = json_to_verts_faces(glob("objs/*.json")[0])
#     mesh = construct_mesh(verts, faces)
#     orb = cv2.ORB_create(nfeatures=1000)
#
#     pts = []
#     xs = np.arange(0, 10, 0.1)
#     for a in xs:
#         renderer = get_renderer(dist=5, elev=30, azim=45 + a)
#         imgs = renderer(mesh)
#
#         imgs = (imgs[..., :3].cpu().numpy() * 255).astype("uint8")
#
#         gray_image = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
#         kp, des = orb.detectAndCompute(gray_image, None)
#         kp_image = cv2.drawKeypoints(imgs[0], kp, None, color=(0, 255, 0), flags=0)
#         # cv2.imshow('ORB', kp_image)
#         # cv2.waitKey(0)
#         points = np.array([k.pt for k in kp])
#         pts.append(points)
#
#     ys = [pt_distance(pts[0], p) for p in pts]
#     plt.plot(xs, ys)
#     plt.show()
