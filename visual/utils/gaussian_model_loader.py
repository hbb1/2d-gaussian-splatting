import os
import glob
import torch
from visual.models.gaussian_model_simplified import GaussianModelSimplified
from visual.renderers.vanilla_renderer import VanillaRenderer


class GaussianModelLoader:
    @staticmethod
    def search_load_file(model_path: str) -> str:
        # if a directory path is provided, auto search checkpoint or ply
        if os.path.isdir(model_path) is False:
            return model_path
        # search checkpoint
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        # find checkpoint with max iterations
        load_from = None
        previous_checkpoint_iteration = -1
        for i in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
            try:
                checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
            except Exception as err:
                print("error occurred when parsing iteration from {}: {}".format(i, err))
                continue
            if checkpoint_iteration > previous_checkpoint_iteration:
                previous_checkpoint_iteration = checkpoint_iteration
                load_from = i

        # not a checkpoint can be found, search point cloud
        if load_from is None:
            previous_point_cloud_iteration = -1
            for i in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
                try:
                    point_cloud_iteration = int(os.path.basename(i).replace("iteration_", ""))
                except Exception as err:
                    print("error occurred when parsing iteration from {}: {}".format(i, err))
                    continue

                if point_cloud_iteration > previous_point_cloud_iteration:
                    previous_point_cloud_iteration = point_cloud_iteration
                    load_from = os.path.join(i, "point_cloud.ply")

        assert load_from is not None, "not a checkpoint or point cloud can be found"

        return load_from

    @staticmethod
    def initialize_simplified_model_from_checkpoint(checkpoint_path: str, device):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]
        sh_degree = hparams["gaussian"].sh_degree

        # initialize gaussian and renderer model
        model = GaussianModelSimplified.construct_from_state_dict(checkpoint["state_dict"], sh_degree, device)
        # extract state dict of renderer
        renderer = hparams["renderer"]
        renderer_state_dict = {}
        for i in checkpoint["state_dict"]:
            if i.startswith("renderer."):
                renderer_state_dict[i[len("renderer."):]] = checkpoint["state_dict"][i]
        # load state dict of renderer
        renderer.load_state_dict(renderer_state_dict)
        renderer = renderer.to(device)

        return model, renderer, checkpoint

    @staticmethod
    def initialize_simplified_model_from_point_cloud(point_cloud_path: str, sh_degree, device):
        model = GaussianModelSimplified.construct_from_ply(ply_path=point_cloud_path, sh_degree=sh_degree, device=device)
        renderer = VanillaRenderer()
        renderer.setup(stage="val")
        renderer = renderer.to(device)

        return model, renderer

    @classmethod
    def search_and_load(cls, model_path: str, sh_degree, device):
        load_from = cls.search_load_file(model_path)
        if load_from.endswith(".ckpt"):
            model, renderer, _ = cls.initialize_simplified_model_from_checkpoint(load_from, device=device)
        elif load_from.endswith(".ply"):
            model, renderer = cls.initialize_simplified_model_from_point_cloud(load_from, sh_degree=sh_degree, device=device)
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer
