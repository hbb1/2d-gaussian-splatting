import numpy as np
import viser
import viser.transforms as vtf
import torch
from visual.dataparsers.colmap_dataparser import ColmapDataParser


class UpDirectionFolder:
    def __init__(self, viewer, server):
        self.viewer = viewer
        self.server = server
        self._setup()

    def _setup(self):
        server = self.server

        # calculate rotation from current up vector
        def calculate_up_rotation():
            rotation_matrix_of_up_direction = ColmapDataParser.rotation_matrix(
                torch.tensor(self.viewer.up_direction / np.linalg.norm(self.viewer.up_direction), dtype=torch.float),
                torch.tensor([0., 0., 1.], dtype=torch.float),
            ).T.numpy()
            return vtf.SO3.from_matrix(rotation_matrix_of_up_direction)


        with server.add_gui_folder("Up Direction"):
            # reset up
            reset_up_button = server.add_gui_button(
                "Reset up direction",
                icon=viser.Icon.ARROW_AUTOFIT_UP,
                hint="Reset the orbit up direction.",
            )
            @reset_up_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                # calculate up vector based on current camera pose
                current_camera_rotation = vtf.SO3(event.client.camera.wxyz)
                new_up_vector = current_camera_rotation @ np.array([0.0, -1.0, 0.0])

                # update client camera
                event.client.camera.up_direction = new_up_vector
                # update text input
                up_direction_vector_input.value = event.client.camera.up_direction
                # update stored value
                self.viewer.up_direction = new_up_vector
                # update orientation control visualizer
                up_direction_visualize_camera.wxyz = event.client.camera.wxyz
                up_rotation = (current_camera_rotation @ vtf.SO3.from_x_radians(np.pi / 2))
                up_direction_visualize_transform.wxyz = up_rotation.wxyz
                # update rotation text inputs
                up_rotations_input.value = up_rotation.as_rpy_radians()


            # up text vector
            up_direction_vector_input = server.add_gui_vector3(
                label="Up",
                initial_value=tuple(self.viewer.up_direction.tolist()),
                step=0.0001,
                disabled=True,
            )

            # setup orientation visualizer (a frustum and transform control)
            rotation_of_up_direction = calculate_up_rotation()
            up_direction_visualize_camera = server.add_camera_frustum(
                "Camera Orientation",
                fov=np.pi / 2,
                aspect=1.5,
                scale=1.,
                color=(1., 0., 1.),
                wxyz=(rotation_of_up_direction @ vtf.SO3.from_x_radians(-np.pi / 2)).wxyz,
                visible=False,
            )
            up_direction_visualize_transform = server.add_transform_controls(
                "Camera Orientation Transform",
                scale=2.,
                wxyz=rotation_of_up_direction.wxyz,
                disable_sliders=True,
                visible=False,
            )
            ## synchronize frustum, transform control and text inputs
            @up_direction_visualize_transform.on_update
            def on_up_direction_visualize_transform_update(event):
                transform_rotation = vtf.SO3(up_direction_visualize_transform.wxyz)
                up_direction_visualize_camera.wxyz = (transform_rotation @ vtf.SO3.from_x_radians(-np.pi / 2)).wxyz
                up_direction_visualize_camera.position = up_direction_visualize_transform.position
                up_rotations_input.value = transform_rotation.as_rpy_radians()

            # toggle the visibilities of visualizers
            show_up_visualizer_checkbox = server.add_gui_checkbox(
                label="Orientation Control",
                initial_value=False,
            )
            @show_up_visualizer_checkbox.on_update
            def _(event):
                # show at where camera look at
                up_direction_visualize_camera.position = event.client.camera.look_at
                up_direction_visualize_camera.visible = show_up_visualizer_checkbox.value
                up_direction_visualize_transform.position = event.client.camera.look_at
                up_direction_visualize_transform.visible = show_up_visualizer_checkbox.value

            # rotations text inputs
            up_rotations_input = server.add_gui_vector3(
                "Up Rot (xyz)",
                initial_value=rotation_of_up_direction.as_rpy_radians(),
                min=(-np.pi, -np.pi, -np.pi),
                max=(np.pi, np.pi, np.pi),
                step=0.01,
            )
            @up_rotations_input.on_update
            def _(event):
                if event.client is None:
                    return
                # synchronize visualizers
                up_direction_visualize_transform.wxyz = vtf.SO3.from_rpy_radians(*up_rotations_input.value).wxyz
                on_up_direction_visualize_transform_update(event)

            # set up direction based on the pose of the visualizer (transform control)
            apply_up_direction = server.add_gui_button("Apply Up Direction")
            @apply_up_direction.on_click
            def _(event):
                new_up_direction = vtf.SO3(up_direction_visualize_transform.wxyz) @ np.array([0.0, 0.0, 1.0])
                self.viewer.up_direction = new_up_direction
                event.client.camera.up_direction = new_up_direction
                up_direction_vector_input.value = new_up_direction
