import os
import torch
import pytorch3d
from pytorch3d.io import IO, load_mesh
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene


device='cpu'
verts, faces = pytorch3d.io.load_ply(os.path.abspath('/home/sneha/airway_exploration/test_tree.ply'))
mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)]
)

# Render the plotly figure
fig = plot_scene({
    "subplot1": {
        "cow_mesh": mesh
    }
})
fig.show()