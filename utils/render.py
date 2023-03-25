import json
import time
from glob import glob

import cv2
import numpy as np
from torch import FloatTensor

import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex, Textures, SoftSilhouetteShader
)
from pytorch3d.structures import Meshes

from utils.utils import get_kpts


def json_to_verts_faces(path):
    with open(path) as json_file:
        data = json.load(json_file)
        vertices = np.array(data['vertices'])
        triangles = np.array(data['faces']) - 1
    return np.array(vertices), np.array(triangles)


def construct_mesh(verts, faces, device="cuda:0"):
    verts, faces = FloatTensor(verts).to(device), FloatTensor(faces).to(device)
    verts[:, 1] *= -1
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    return Meshes(verts=[verts], faces=[faces], textures=textures)


def get_renderer(dist=10, elev=0, azim=180, imsize=512, device="cuda:0"):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=imsize,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    cam_pos = cameras[0].get_camera_center()
    lights = PointLights(device=device, location=-cam_pos)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


if __name__ == '__main__':
    dist, elev, azim, class_id = 10, 6, 40, 3
    # dist, elev, azim, class_id = [ 5.078, 33.075, 46.074,  5.187]

    verts, faces = json_to_verts_faces(glob("data/objs/*.json")[round(class_id)])
    mesh = construct_mesh(verts, faces)

    renderer = get_renderer(dist=dist, elev=elev, azim=azim, imsize=512)
    img = renderer(mesh)[0, ..., :3].detach().cpu().numpy()
    kpts = get_kpts(renderer, mesh)
    for kpt in kpts:
        cv2.circle(img, (int(kpt[0]), int(kpt[1])), 2, (0, 0, 255), -1)
    cv2.imshow("img", img)

    cv2.waitKey(0)
