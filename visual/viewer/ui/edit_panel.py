import traceback
import datetime
import os.path

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re


class EditPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab

        self._setup_point_cloud_folder()
        self._setup_gaussian_edit_folder()
        self._setup_save_gaussian_folder()

    def _setup_point_cloud_folder(self):
        server = self.server
        with self.server.add_gui_folder("Point Cloud"):
            self.show_point_cloud_checkbox = server.add_gui_checkbox(
                "Show Point Cloud",
                initial_value=False,
            )
            self.point_cloud_color = server.add_gui_vector3(
                "Point Color",
                min=(0, 0, 0),
                max=(255, 255, 255),
                step=1,
                initial_value=(0, 255, 255),
            )
            self.point_size = server.add_gui_number(
                "Point Size",
                min=0.,
                initial_value=0.01,
            )
            self.point_sparsify = server.add_gui_number(
                "Point Sparsify",
                min=1,
                initial_value=10,
            )

            self.pcd = None

            @self.show_point_cloud_checkbox.on_update
            @self.point_cloud_color.on_update
            @self.point_size.on_update
            @self.point_sparsify.on_update
            def _(event: viser.GuiEvent):
                with self.server.atomic():
                    self._update_pcd()

    def _resize_grid(self, idx):
        exist_grid = self.grids[idx][0]
        exist_grid.remove()
        self.grids[idx][0] = self.server.add_grid(
            "/grid/{}".format(idx),
            width=self.grids[idx][2].value[0],
            height=self.grids[idx][2].value[1],
            wxyz=self.grids[idx][1].wxyz,
            position=self.grids[idx][1].position,
        )
        self._update_scene()

    def _setup_gaussian_edit_folder(self):
        server = self.server

        self.edit_histories = []

        with server.add_gui_folder("Edit"):
            # initialize a list to store panel(grid)'s information
            self.grids: dict[int, list[
                viser.MeshHandle,
                viser.TransformControlsHandle,
                viser.GuiInputHandle,
            ]] = {}
            self.grid_idx = 0

            add_grid_button = server.add_gui_button("Add Panel")
            self.delete_gaussians_button = server.add_gui_button(
                "Delete Gaussians",
                color="red",
            )

        self.grid_folders = {}

        # create panel(grid)
        def new_grid(idx):
            with self.server.add_gui_folder("Grid {}".format(idx)) as folder:
                self.grid_folders[idx] = folder

                # TODO: add height
                grid_size = server.add_gui_vector2("Size", initial_value=(10., 10.), min=(0., 0.), step=0.01)

                grid = server.add_grid(
                    "/grid/{}".format(idx),
                    height=grid_size.value[0],
                    width=grid_size.value[1],
                )
                grid_transform = server.add_transform_controls(
                    "/grid_transform_control/{}".format(idx),
                    wxyz=grid.wxyz,
                    position=grid.position,
                )

                # resize panel on size value changed
                @grid_size.on_update
                def _(event: viser.GuiEvent):
                    with event.client.atomic():
                        self._resize_grid(idx)

                # handle panel deletion
                grid_delete_button = server.add_gui_button("Delete")

                @grid_delete_button.on_click
                def _(_):
                    with server.atomic():
                        try:
                            self.grids[idx][0].remove()
                            self.grids[idx][1].remove()
                            self.grids[idx][2].remove()
                            self.grid_folders[idx].remove()  # bug
                        except Exception as e:
                            traceback.print_exc()
                        finally:
                            del self.grids[idx]
                            del self.grid_folders[idx]

                    self._update_scene()

            # update the pose of panel(grid) when grid_transform updated
            @grid_transform.on_update
            def _(_):
                self.grids[idx][0].wxyz = grid_transform.wxyz
                self.grids[idx][0].position = grid_transform.position
                self._update_scene()

            self.grids[self.grid_idx] = [grid, grid_transform, grid_size]
            self._update_scene()

        # setup callbacks

        @add_grid_button.on_click
        def _(_):
            with server.atomic():
                new_grid(self.grid_idx)
                self.grid_idx += 1

        @self.delete_gaussians_button.on_click
        def _(_):
            with server.atomic():
                gaussian_to_be_deleted, pose_and_size_list = self._get_selected_gaussians_mask(return_pose_and_size_list=True)
                self.edit_histories.append(pose_and_size_list)
                self.viewer.gaussian_model.delete_gaussians(gaussian_to_be_deleted)
                self._update_pcd()
            self.viewer.rerender_for_all_client()

    def _setup_save_gaussian_folder(self):
        with self.server.add_gui_folder("Save"):
            name_text = self.server.add_gui_text(
                "Name",
                initial_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            save_button = self.server.add_gui_button("Save")

            @save_button.on_click
            def _(event: viser.GuiEvent):
                # skip if not triggered by client
                if event.client is None:
                    return
                try:
                    save_button.disabled = True

                    with self.server.atomic():
                        try:
                            # check whether is a valid name
                            name = name_text.value
                            match = re.search("^[a-zA-Z0-9_\-]+$", name)
                            if match:
                                output_directory = "edited"
                                os.makedirs(output_directory, exist_ok=True)
                                try:
                                    if len(self.edit_histories) > 0:
                                        torch.save(self.edit_histories, os.path.join(output_directory, f"{name}-edit_histories.ckpt"))
                                except:
                                    traceback.print_exc()

                                if self.viewer.checkpoint is None:
                                    # save ply
                                    ply_save_path = os.path.join(output_directory, "{}.ply".format(name))
                                    self.viewer.gaussian_model.to_ply_structure().save_to_ply(ply_save_path)
                                    message_text = "Saved to {}".format(ply_save_path)
                                else:
                                    # save as a checkpoint if viewer started from a checkpoint
                                    checkpoint_save_path = os.path.join(output_directory, "{}.ckpt".format(name))
                                    checkpoint = self.viewer.checkpoint
                                    # update state dict of the checkpoint
                                    state_dict_value = self.viewer.gaussian_model.to_parameter_structure()
                                    for name_in_dict, name_in_dataclass in [
                                        ("xyz", "xyz"),
                                        ("features_dc", "features_dc"),
                                        ("features_rest", "features_rest"),
                                        ("scaling", "scales"),
                                        ("rotation", "rotations"),
                                        ("opacity", "opacities"),
                                        ("features_extra", "real_features_extra"),
                                    ]:
                                        dict_key = "gaussian_model._{}".format(name_in_dict)
                                        if dict_key not in checkpoint["state_dict"]:
                                            print(f"WARNING: `{dict_key}` not found in original checkpoint")
                                        checkpoint["state_dict"][dict_key] = getattr(state_dict_value, name_in_dataclass)
                                    # save
                                    torch.save(checkpoint, checkpoint_save_path)
                                    message_text = "Saved to {}".format(checkpoint_save_path)
                            else:
                                message_text = "Invalid name"
                        except:
                            traceback.print_exc()

                    # show message
                    with event.client.add_gui_modal("Message") as modal:
                        event.client.add_gui_markdown(message_text)
                        close_button = event.client.add_gui_button("Close")

                        @close_button.on_click
                        def _(_) -> None:
                            modal.close()

                finally:
                    save_button.disabled = False

    def _get_selected_gaussians_mask(self, return_pose_and_size_list: bool = False):
        xyz = self.viewer.gaussian_model.get_xyz

        # if no grid exists, do not delete any gaussians
        if len(self.grids) == 0:
            return torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.bool)

        pose_and_size_list = []
        # initialize mask with True
        is_gaussian_selected = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.bool)
        for i in self.grids:
            # get the pose of grid, and build world-to-grid transform matrix
            grid = self.grids[i][0]
            se3 = torch.linalg.inv(torch.tensor(vtf.SE3.from_rotation_and_translation(
                vtf.SO3(grid.wxyz),
                grid.position,
            ).as_matrix()).to(xyz))
            # transform xyz from world to grid
            new_xyz = torch.matmul(xyz, se3[:3, :3].T) + se3[:3, 3]
            # find the gaussians to be deleted based on the new_xyz
            grid_size = self.grids[i][2].value
            x_mask = torch.abs(new_xyz[:, 0]) < grid_size[0] / 2
            y_mask = torch.abs(new_xyz[:, 1]) < grid_size[1] / 2
            z_mask = new_xyz[:, 2] > 0
            # update mask
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, x_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, y_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, z_mask)

            # add to history
            pose_and_size_list.append((se3.cpu(), grid_size))

        if return_pose_and_size_list is True:
            return is_gaussian_selected, pose_and_size_list
        return is_gaussian_selected

    def _get_selected_gaussians_indices(self):
        """
        get the index of the gaussians which in the range of grids
        :return:
        """
        selected_gaussian = torch.where(self._get_selected_gaussians_mask())
        return selected_gaussian

    def _update_pcd(self, selected_gaussians_indices=None):
        self.remove_point_cloud()
        if self.show_point_cloud_checkbox.value is False:
            return
        xyz = self.viewer.gaussian_model.get_xyz
        colors = torch.tensor([self.point_cloud_color.value], dtype=torch.uint8, device=xyz.device).repeat(xyz.shape[0], 1)
        if selected_gaussians_indices is None:
            selected_gaussians_indices = self._get_selected_gaussians_indices()
        colors[selected_gaussians_indices] = 255 - torch.tensor(self.point_cloud_color.value).to(colors)

        point_sparsify = int(self.point_sparsify.value)
        self.show_point_cloud(xyz[::point_sparsify].cpu().numpy(), colors[::point_sparsify].cpu().numpy())

    def remove_point_cloud(self):
        if self.pcd is not None:
            self.pcd.remove()
            self.pcd = None

    def show_point_cloud(self, xyz, colors):
        self.pcd = self.server.add_point_cloud(
            "/pcd",
            points=xyz,
            colors=colors,
            point_size=self.point_size.value,
        )

    def _update_scene(self):
        selected_gaussians_indices = self._get_selected_gaussians_mask()
        self.viewer.gaussian_model.select(selected_gaussians_indices)
        self._update_pcd(selected_gaussians_indices)

        self.viewer.rerender_for_all_client()
