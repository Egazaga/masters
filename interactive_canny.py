from glob import glob
import cv2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from pytorch3d.renderer import PointLights, FoVPerspectiveCameras, look_at_view_transform

from utils.render import construct_mesh, json_to_verts_faces, get_renderer
from utils.model_subset import get_models_subset


def get_imgs(meshes, renderer, dist, elev, azim, fov):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)  # , up=((-0.1, 1, 0),)
    camera = FoVPerspectiveCameras(device="cuda:0", R=R, T=T, fov=fov)
    d = 5.0
    light_coords = [[0.0, 0.0, -d], [0.0, 0.0, d], [0.0, -d, 0.0], [0.0, d, 0.0], [-d, 0.0, 0.0], [d, 0.0, 0.0]]
    lights = PointLights(device="cuda:0", location=light_coords)
    meshes_ext = meshes.extend(len(light_coords))
    imgs = renderer(meshes_ext, lights=lights, cameras=camera)[..., :3].detach().cpu().numpy()

    # apply canny
    cannys = []
    for i in range(len(imgs)):
        cannys.append(cv2.Canny((imgs[i] * 255).astype(np.uint8), 100, 200))
    img2 = np.max(cannys, axis=0)
    return imgs[2], img2


if __name__ == '__main__':
    dist, elev, azim, fov, class_id = 6, 7, 45, 50, 3

    subset = get_models_subset(20)
    meshes = [construct_mesh(*json_to_verts_faces(path)) for i, path in enumerate(glob("data/objs/*.json")) if
              i in subset][class_id]

    renderer = get_renderer(dist=dist, elev=elev, azim=azim, imsize=512)
    img1, img2 = get_imgs(meshes, renderer, dist, elev, azim, fov)

    # Create the figure and subplots
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Plot the images
    ax[0].imshow(img1)
    ax[1].imshow(img2)

    # Set the axes limits and aspect ratios
    ax[0].set_xlim([0, img1.shape[1]])
    ax[0].set_ylim([img1.shape[0], 0])
    ax[0].set_aspect('equal')
    ax[1].set_xlim([0, img1.shape[1]])
    ax[1].set_ylim([img1.shape[0], 0])
    ax[1].set_aspect('equal')

    photo = cv2.cvtColor(cv2.imread("data/audi.png"), cv2.COLOR_BGR2RGB)
    ax2[0].imshow(photo)
    ax2[1].imshow(cv2.Canny(photo, 100, 200))

    # Create the distance slider
    ax_dist = plt.axes([0.1, 0.05, 0.8, 0.02])
    slider_dist = Slider(ax_dist, 'Dist', 0.1, 10.0, valinit=dist)

    # Create the elevation slider
    ax_elev = plt.axes([0.1, 0.02, 0.8, 0.02])
    slider_elev = Slider(ax_elev, 'Elev', 0.0, 90.0, valinit=elev)

    # Create the azimuth slider
    ax_azim = plt.axes([0.1, 0.08, 0.8, 0.02])
    slider_azim = Slider(ax_azim, 'Azim', -180.0, 180.0, valinit=azim)

    # Create the fov slider
    ax_fov = plt.axes([0.1, 0.11, 0.8, 0.02])
    slider_fov = Slider(ax_fov, 'Fov', 20, 90, valinit=fov)


    # Define the update function for the sliders
    def update(val):
        # Get the current slider values
        dist = slider_dist.val
        elev = slider_elev.val
        azim = slider_azim.val
        fov = slider_fov.val
        # New FOV = 2 x arctan((0.5 x Object Size) / New Distance to Object)
        new_fov = 2 * np.arctan(0.5 * 6 / dist) * 180 / np.pi
        # slider_fov.set_val(new_fov)
        print(new_fov)

        img1, img2 = get_imgs(meshes, renderer, dist, elev, azim, new_fov)
        ax[0].imshow(img1)
        ax[1].imshow(img2)

        # Redraw the figure
        fig.canvas.draw_idle()


    # Connect the sliders to the update function
    slider_dist.on_changed(update)
    slider_elev.on_changed(update)
    slider_azim.on_changed(update)
    slider_fov.on_changed(update)

    # Show the figure
    plt.show()
