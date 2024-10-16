from gaustudio.pipelines import initializers, optimizers
from gaustudio import models

bg_gaussians_coarse = models.make("vanilla_pcd")
bg_initializer = initializers.make({"name": "gaussiansky", "radius": 100, "resolution": 500})
bg_initializer(bg_gaussians_coarse)
bg_gaussians_coarse.export("assets/background_gs.ply")