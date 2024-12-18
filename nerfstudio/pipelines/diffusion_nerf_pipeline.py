# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Literal, Optional, Tuple, Type, Any, Union, cast
import random
from copy import deepcopy
import typing

from pydantic import BaseModel
import torch
from torch import Tensor, FloatTensor
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_pil_image, to_tensor

from nerfstudio.data.datamanagers.ad_datamanager import (
    ADDataManager,
    ADDataManagerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.generative.diffusion_model import (
    StableDiffusionModel,
    DiffusionModelConfig,
)
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras


import math

DEG_TO_RAD = math.pi / 180


class PoseConfig(BaseModel):
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0

    @property
    def tensor_deg(self) -> Tensor:
        return torch.tensor(
            [self.pos_x, self.pos_y, self.pos_z, self.rot_x, self.rot_y, self.rot_z]
        )

    @property
    def tensor_rad(self) -> Tensor:
        return torch.tensor(
            [
                self.pos_x,
                self.pos_y,
                self.pos_z,
                self.rot_x * DEG_TO_RAD,
                self.rot_y * DEG_TO_RAD,
                self.rot_z * DEG_TO_RAD,
            ]
        )


@dataclass
class DiffusionNerfConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DiffusionNerfPipeline)
    """target class to instantiate"""
    datamanager: ADDataManagerConfig = field(default_factory=ADDataManagerConfig)
    """specifies the datamanager config"""
    model: ADModelConfig = field(default_factory=ADModelConfig)
    """specifies the model config"""

    nerf_checkpoint: Optional[str] = None
    """Checkpoint path of the NeRF model."""

    note: str = ""
    """Note to add to the experiment name."""

    calc_fid_steps: Tuple[int, ...] = (
        5000,
        10000,
        15000,
        20000,
    )  # NOTE: must also be an eval step for this to work

    eval_shift_distances_horizontal: Tuple[int, ...] = (0, 2, 4, 6, 8)
    eval_shift_distances_vertical: Tuple[int, ...] = (1,)
    """Whether to calculate FID for lane shifted images."""
    ray_patch_size: Tuple[int, int] = (128, 128)
    """Size of the ray patches to sample from the image during training (for camera rays only)."""

    diffusion_model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    """Configuration for the diffusion model used for augmentation."""

    diffusion_seed: Optional[None] = None

    augment_phase_step: int = 0
    max_aug_phase_step: int = 10000
    noise_start_phase_step: int = 10000
    noise_end_phase_step: int = 20001

    augment_loss_mult: float = 1.0
    augment_strategy: Literal["none", "partial_const", "partial_linear"] = (
        "partial_const"
    )
    """ Which diffusion augmentation strategy to use.
        Can choose between `none`, `partial_const`, `partial_linear`.
    """

    augment_max_strength: PoseConfig = PoseConfig(
        pos_x=5.0,
        pos_y=0,
        pos_z=0,
        rot_x=0,
        rot_y=0,
        rot_z=45,
    )
    """The range in which shifts and rotations get uniformly sampled from. (-x, x)"""

    augment_direction: PoseConfig = PoseConfig(
        pos_x=-1,
        pos_y=1,
        pos_z=1,
        rot_x=-1,
        rot_y=1,
        rot_z=1,
    )

    max_steps: int = 20001
    debug_image_interval: int = 10

    def __post_init__(self) -> None:
        assert (
            self.ray_patch_size[0] == self.ray_patch_size[1]
        ), "Non-square patches are not supported yet, sorry."
        self.datamanager.image_divisible_by = self.model.rgb_upsample_factor


class DiffusionNerfPipeline(VanillaPipeline):
    """Pipeline for training AD models."""

    def __init__(self, config: DiffusionNerfConfig, **kwargs):
        pixel_sampler = config.datamanager.pixel_sampler
        pixel_sampler.patch_size = config.ray_patch_size[0]
        pixel_sampler.patch_scale = config.model.rgb_upsample_factor
        super().__init__(config, **kwargs)

        # Fix type hints
        self.datamanager: ADDataManager = self.datamanager
        self.model: ADModel = self.model
        self.config: DiffusionNerfConfig = self.config
        self.diffusion_model = self.config.diffusion_model.setup(
            device=self.device,
        )

        # Disable ray drop classification if we do not add missing points
        if not self.datamanager.dataparser.config.add_missing_points:
            self.model.disable_ray_drop()

        self.fid = None

        if (
            self.config.nerf_checkpoint
            and self.config.nerf_checkpoint.strip().lower() != "none"
        ):
            self.load_nerf_checkpoint(self.config.nerf_checkpoint)

    def load_nerf_checkpoint(self, checkpoint_path: Union[Path, str]):
        checkpoint_path = Path(checkpoint_path)

        with open(self.config.nerf_checkpoint, "rb") as f:
            loaded_state_dict = torch.load(f, map_location="cpu")["pipeline"]

        loaded_state_dict = {
            key.replace("module.", ""): value
            for key, value in loaded_state_dict.items()
        }

        self.load_state_dict(loaded_state_dict, strict=True)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        # Regular forward pass and loss calc
        self.train()
        ray_bundle, batch = self.datamanager.next_train(step)
        cameras = self.datamanager.train_dataset.cameras

        if self.config.augment_strategy == "none":
            return self._strategy_augment_none(ray_bundle, batch, use_actor_shift=True)
        elif self.config.augment_strategy in ["partial_const", "partial_linear"]:
            return self._strategy_augment_partial_const(
                ray_bundle,
                batch,
                step,
                cameras,
                use_actor_shift=True,
            )
        else:
            raise ValueError(
                "Unrecognized augment strategy", self.config.augment_strategy
            )

    def _strategy_augment_none(self, ray_bundle, batch, use_actor_shift=True):
        model_outputs = self._model(
            deepcopy(ray_bundle), patch_size=self.config.ray_patch_size
        )
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if (
            use_actor_shift
            and (actors := self.model.dynamic_actors).config.optimize_trajectories
        ):
            pos_norm = (actors.actor_positions - actors.initial_positions).norm(dim=-1)
            metrics_dict["traj_opt_translation"] = (
                pos_norm[pos_norm > 0].mean().nan_to_num()
            )
            metrics_dict["traj_opt_rotation"] = (
                (actors.actor_rotations_6d - actors.initial_rotations_6d)[pos_norm > 0]
                .norm(dim=-1)
                .mean()
                .nan_to_num()
            )
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def _strategy_augment_partial_const(
        self,
        ray_bundle: RayBundle,
        batch,
        step: int,
        cameras: Cameras,
        use_actor_shift: bool = False,
    ):
        model_outputs, loss_dict, metrics_dict = self._strategy_augment_none(
            ray_bundle.to(ray_bundle.origins.device), batch, use_actor_shift
        )
        model_outputs = nan_to_num_batch(model_outputs)

        if not self._is_augment_phase(step):
            return model_outputs, loss_dict, metrics_dict

        pose_aug = self._get_pose_augmentation(step)
        aug_ray_bundle = transform_ray_bundle(ray_bundle, pose_aug, cameras)

        # with torch.autograd.detect_anomaly():
        aug_outputs = self.model(aug_ray_bundle, patch_size=self.config.ray_patch_size)
        aug_outputs = nan_to_num_batch(aug_outputs)

        assert len(aug_outputs["rgb"].shape) == 4
        diffusion_input = {"rgb": aug_outputs["rgb"].permute(0, 3, 1, 2)}

        for signal_name in self.diffusion_model.config.conditioning_signals:
            if signal_name == "ray":
                diffusion_input[signal_name] = get_ray_signal_from_ray_bundle(
                    aug_ray_bundle,
                    self.config.ray_patch_size,
                    self.config.model.rgb_upsample_factor,
                )

            else:
                raise NotImplementedError

        diffusion_model = cast(StableDiffusionModel, self.diffusion_model)

        with torch.no_grad():
            diffusion_output = diffusion_model.get_diffusion_output(
                diffusion_input,
                strength=self._get_diffusion_strength(step),
                seed=self.config.diffusion_seed,
            )
        diffusion_output = nan_to_num_batch(diffusion_output)

        aug_metrics_dict = diffusion_model.get_diffusion_metrics(
            diffusion_input, diffusion_output
        )
        aug_loss_dict = diffusion_model.get_diffusion_losses(
            diffusion_input, diffusion_output
        )
        aug_loss_dict = {
            k: v * self.config.augment_loss_mult for k, v in aug_loss_dict.items()
        }

        if (
            (self.config.debug_image_interval is not None)
            and (step > 0)
            and (step % self.config.debug_image_interval == 0)
        ):
            image_dict = {
                "gt": batch["image"],
                "nerf": model_outputs["rgb"],
                "aug": aug_outputs["rgb"],
                "diffusion": diffusion_output["rgb"].permute(0, 2, 3, 1),
            }
            for image_name, image in image_dict.items():
                writer.put_image(
                    name="Debug Images" + "/" + image_name,
                    image=image[0].detach().float().cpu(),
                    step=step,
                )

        model_outputs.update({("aug_" + k): v for k, v in aug_outputs.items()})
        loss_dict.update({("aug_" + k): v for k, v in aug_loss_dict.items()})
        metrics_dict.update({("aug_" + k): v for k, v in aug_metrics_dict.items()})

        return model_outputs, loss_dict, metrics_dict

    def _get_pose_augmentation(self, step, event: Optional[Tensor] = None) -> Tensor:
        return get_pose_augmentation(
            step,
            self.config.augment_max_strength,
            self.config.augment_direction,
            self.config.max_aug_phase_step,
            event,
            self.config.augment_strategy,
        )

    def _get_diffusion_strength(self, step: int) -> float:
        return get_diffusion_strength(
            step,
            self.config.diffusion_model.noise_strength,
            self.config.noise_start_phase_step,
            self.config.noise_end_phase_step,
            self.config.diffusion_model.num_inference_steps,
        )

    def _is_augment_phase(self, step: int) -> bool:
        return step >= self.config.augment_phase_step

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()

        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, patch_size=self.config.ray_patch_size)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()

        # Image eval
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(
            outputs, batch
        )
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()

        # Lidar eval
        lidar, batch = self.datamanager.next_eval_lidar(step)
        outputs, batch = self.model.get_outputs_for_lidar(lidar, batch=batch)
        lidar_metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
        assert not set(lidar_metrics_dict.keys()).intersection(metrics_dict.keys())
        metrics_dict.update(lidar_metrics_dict)

        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

         Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """

        # Taken and modified from the original AD pipeline

        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            lane_shift_fids = (
                {
                    i: FrechetInceptionDistance().to(self.device)
                    for i in self.config.eval_shift_distances_horizontal
                }
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            vertical_shift_fids = (
                {
                    i: FrechetInceptionDistance().to(self.device)
                    for i in self.config.eval_shift_distances_vertical
                }
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            actor_edits = {
                "rot": [(0.5, 0), (-0.5, 0)],
                "trans": [(0, 2.0), (0, -2.0)],
                # "both": [(0.5, 2.0), (-0.5, 2.0), (0.5, -2.0), (-0.5, -2.0)],
            }
            actor_fids = (
                {
                    k: FrechetInceptionDistance().to(self.device)
                    for k in actor_edits.keys()
                }
                if step in self.config.calc_fid_steps or step is None
                else {}
            )
            if actor_fids:
                actor_fids["true"] = FrechetInceptionDistance().to(self.device)

            num_images = len(self.datamanager.fixed_indices_eval_dataloader)
            task = progress.add_task(
                "[green]Evaluating all eval images...", total=num_images
            )
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                # Generate images from the original rays
                camera_ray_bundle = camera.generate_rays(0, keep_shape=True)
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle
                )
                # Compute metrics for the original image
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(
                    outputs, batch
                )
                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path
                            / "{0:06d}-{1}.jpg".format(
                                int(camera_indices[0, 0, 0]), key
                            )
                        )
                # Add timing stuff
                assert "num_rays_per_sec" not in metrics_dict
                num_rays = math.prod(camera_ray_bundle.shape)
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / num_rays
                metrics_dict_list.append(metrics_dict)
                if lane_shift_fids:
                    self._update_lane_shift_fid(
                        lane_shift_fids,
                        camera_ray_bundle,
                        batch["image"],
                        outputs["rgb"],
                    )
                if vertical_shift_fids:
                    self._update_vertical_shift_fid(
                        vertical_shift_fids, camera_ray_bundle, batch["image"]
                    )
                if actor_fids:
                    self._update_actor_fids(
                        actor_fids, actor_edits, camera_ray_bundle, batch["image"]
                    )
                progress.advance(task)
            num_lidar = len(self.datamanager.fixed_indices_eval_lidar_dataloader)
            task = progress.add_task(
                "[green]Evaluating all eval point clouds...", total=num_lidar
            )
            for lidar, batch in self.datamanager.fixed_indices_eval_lidar_dataloader:
                outputs, batch = self.model.get_outputs_for_lidar(lidar, batch=batch)
                metrics_dict, _ = self.model.get_image_metrics_and_images(
                    outputs, batch
                )
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        # average the metrics list
        metrics_dict = {}
        keys = {
            key for metrics_dict in metrics_dict_list for key in metrics_dict.keys()
        }
        # remove the keys related to actor metrics as they need to be averaged differently
        actor_keys = {key for key in keys if key.startswith("actor_")}
        keys = keys - actor_keys

        for key in keys:
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [
                            metrics_dict[key]
                            for metrics_dict in metrics_dict_list
                            if key in metrics_dict
                        ]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [
                                metrics_dict[key]
                                for metrics_dict in metrics_dict_list
                                if key in metrics_dict
                            ]
                        )
                    )
                )
        # average the actor metrics. Note that due to the way we compute the actor metrics,
        # we need to weight them by how big portion of the image they cover.
        actor_metrics_dict = [md for md in metrics_dict_list if "actor_coverage" in md]
        if actor_metrics_dict:
            actor_coverages = torch.tensor(
                [md["actor_coverage"] for md in actor_metrics_dict]
            )
            for key in actor_keys:
                # we dont want to average the actor coverage in this way.
                if key == "actor_coverage":
                    continue
                # we should weight the actor metrics by the actor coverage
                metrics_dict[key] = float(
                    torch.sum(
                        torch.tensor(
                            [md[key] for md in actor_metrics_dict],
                        )
                        * actor_coverages
                    )
                    / actor_coverages.sum()
                )

        # Add FID metrics (if applicable)
        for shift, fid in lane_shift_fids.items():
            metrics_dict[f"lane_shift_{shift}_fid"] = fid.compute().item()

        for shift, fid in vertical_shift_fids.items():
            metrics_dict[f"vertical_shift_{shift}_fid"] = fid.compute().item()

        if actor_fids:
            for edit_type in actor_edits.keys():
                metrics_dict[f"actor_shift_{edit_type}_fid"] = (
                    actor_fids[edit_type].compute().item()
                )

        return metrics_dict

    @staticmethod
    def _downsample_img(
        img: torch.Tensor,
        out_size: Tuple[int, int] = (
            299,
            299,
        ),
    ):
        """Converts tensor to PIL, downsamples with bicubic, and converts back to tensor."""
        img = to_pil_image(img)
        img = img.resize(out_size, Image.BICUBIC)
        img = to_tensor(img)
        return img

    def _update_lane_shift_fid(
        self, fids: Dict[int, FrechetInceptionDistance], ray_bundle, orig_img, gen_img
    ):
        """Updates the FID metrics (for shifted views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255)
            .unsqueeze(0)
            .to(torch.uint8)
            .to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # TODO: Replace this logic with augment_ray_bundle
        driving_direction = ray_bundle.metadata["velocities"][0, 0, :]
        driving_direction = driving_direction / driving_direction.norm()
        orth_right_direction = torch.cross(
            driving_direction,
            torch.tensor([0.0, 0.0, 1.0], device=driving_direction.device),
        )
        shift_sign = self.datamanager.eval_lidar_dataset.metadata.get(
            "lane_shift_sign", 1
        )  # TODO: Do we need to take z axis into account?
        shift_direction = orth_right_direction * shift_sign

        # Compute FID for shifted views
        imgs_generated = {0: gen_img}
        original_ray_origins = ray_bundle.origins.clone()
        for dist in filter(lambda d: d != 0, fids.keys()):
            shifted_origins = original_ray_origins + dist * shift_direction
            ray_bundle.origins = shifted_origins
            imgs_generated[dist] = self.model.get_outputs_for_camera_ray_bundle(
                ray_bundle
            )["rgb"]
        ray_bundle.origins = original_ray_origins

        for shift, img in imgs_generated.items():
            img = (
                (self._downsample_img((img).permute(2, 0, 1)) * 255)
                .unsqueeze(0)
                .to(torch.uint8)
                .to(self.device)
            )
            fids[shift].update(img, real=False)

    def _update_vertical_shift_fid(
        self, fids: Dict[int, FrechetInceptionDistance], ray_bundle, orig_img
    ):
        """Updates the FID metrics (for shifted views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255)
            .unsqueeze(0)
            .to(torch.uint8)
            .to(self.device)
        )
        fids_list = list(fids.values())
        fids_list[0].update(img_original, real=True)
        for fid in fids_list[1:]:
            fid.real_features_sum = fids_list[0].real_features_sum
            fid.real_features_cov_sum = fids_list[0].real_features_cov_sum
            fid.real_features_num_samples = fids_list[0].real_features_num_samples

        # TODO: Replace this logic with augment_ray_bundle
        shift_direction = torch.tensor(
            [0.0, 0.0, 1.0], device=ray_bundle.origins.device
        )

        # Compute FID for shifted views
        imgs_generated = {}
        original_ray_origins = ray_bundle.origins.clone()
        for dist in filter(lambda d: d != 0, fids.keys()):
            shifted_origins = original_ray_origins + dist * shift_direction
            ray_bundle.origins = shifted_origins
            imgs_generated[dist] = self.model.get_outputs_for_camera_ray_bundle(
                ray_bundle
            )["rgb"]
        ray_bundle.origins = original_ray_origins

        for shift, img in imgs_generated.items():
            img = (
                (self._downsample_img((img).permute(2, 0, 1)) * 255)
                .unsqueeze(0)
                .to(torch.uint8)
                .to(self.device)
            )
            fids[shift].update(img, real=False)

    def _update_actor_fids(
        self,
        fids: Dict[str, FrechetInceptionDistance],
        actor_edits: Dict[str, List[Tuple]],
        ray_bundle,
        orig_img,
    ) -> None:
        """Updates the FID metrics (for shifted actor views) for the given ray bundle and images."""
        # Update "true" FID (with hack to only compute it once)
        img_original = (
            (self._downsample_img((orig_img).permute(2, 0, 1)) * 255)
            .unsqueeze(0)
            .to(torch.uint8)
            .to(self.device)
        )
        fids["true"].update(img_original, real=True)
        for edit_type in actor_edits.keys():
            fids[edit_type].real_features_sum = fids["true"].real_features_sum
            fids[edit_type].real_features_cov_sum = fids["true"].real_features_cov_sum
            fids[edit_type].real_features_num_samples = fids[
                "true"
            ].real_features_num_samples

        # Compute FID for actor edits
        imgs_generated_per_edit = {}
        for edit_type in actor_edits.keys():
            imgs = []
            for rotation, lateral in actor_edits[edit_type]:
                self.model.dynamic_actors.actor_editing["rotation"] = rotation
                self.model.dynamic_actors.actor_editing["lateral"] = lateral
                imgs.append(
                    self.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"]
                )
            imgs_generated_per_edit[edit_type] = imgs

        for edit_type, imgs in imgs_generated_per_edit.items():
            for img in imgs:
                img = (
                    (self._downsample_img((img).permute(2, 0, 1)) * 255)
                    .unsqueeze(0)
                    .to(torch.uint8)
                    .to(self.device)
                )
                fids[edit_type].update(img, real=False)

        self.model.dynamic_actors.actor_editing["rotation"] = 0
        self.model.dynamic_actors.actor_editing["lateral"] = 0


def rotate_around(theta, dim: int, device=None) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)

    if dim == 0:
        r = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif dim == 1:
        r = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    elif dim == 2:
        r = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    else:
        raise ValueError

    return torch.tensor(r, device=device, dtype=torch.float32)


def transform_ray_bundle(
    ray_bundle: RayBundle, pose_offset: Tensor, cameras: Cameras
) -> RayBundle:
    new_ray_bundle = deepcopy(
        ray_bundle
    )  # In case ground truth needs original ray_bundle
    device = new_ray_bundle.origins.device

    aug_translation = pose_offset[..., :3].to(device=device)  # 3
    aug_rotation = pose_offset[..., 3:].to(device=device)  # 3

    is_cam = ~ray_bundle.metadata["is_lidar"].flatten()  # B, 1

    cam_idxs = ray_bundle.camera_indices[is_cam, 0].cpu()  # Bc
    c2w = cameras.camera_to_worlds[cam_idxs].to(device=device)  # Bc, 3, 4

    translation = c2w[..., :3] @ aug_translation

    # opencv_to_nerfstudio = torch.from_numpy(OPENCV_TO_NERFSTUDIO).to(
    #    dtype=torch.float32, device=c2w.device
    # )
    # c2w_rot = c2w[..., :3, :3]

    local_rotation = (  # Chain together rotations, X -> Y -> Z
        rotate_around(aug_rotation[2], 2)
        @ rotate_around(aug_rotation[1], 1)
        @ rotate_around(aug_rotation[0], 0)
    ).to(device)

    # Each direction is in world space
    # We want to apply the rotation in camera space and then transform back to world space
    # We first transform to camera space with c2w^-1
    # Then apply the local rotation in camera space
    # Then transform back to world space with c2w
    # Since matrix multiplication is associative, we can combine these into one matrix
    # We then use einsum to apply multiplication across batches
    # direction_transform = c2w_rot @ local_rotation @ torch.inverse(c2w_rot)

    direction = torch.einsum(
        "bij,bj->bi", local_rotation[None, ...], ray_bundle.directions[is_cam]
    )

    new_ray_bundle.origins[is_cam] += translation.to(new_ray_bundle.origins.dtype)
    new_ray_bundle.directions[is_cam] = direction.to(new_ray_bundle.directions.dtype)

    return new_ray_bundle


def nan_to_num_batch(
    batch: Dict[str, Any], nan: float = 0, posinf: float = 1, neginf: float = 0
) -> Dict[str, Any]:
    for k, v in batch.items():
        if (v is not None) and isinstance(v, Tensor):
            batch[k] = v.nan_to_num(posinf=posinf, neginf=neginf, nan=nan)
    return batch


def is_cam_ray(ray_bundle: RayBundle) -> Tensor:
    return ~ray_bundle.metadata["is_lidar"].flatten()


def unfold_ray_vec(ray: Tensor, patch_size: Tuple[int, int]) -> Tensor:
    assert len(ray.shape) == 2
    assert ray.shape[0] % (patch_size[0] * patch_size[1]) == 0

    C = ray.shape[1]
    B = ray.shape[0] // (patch_size[0] * patch_size[1])
    return ray.view(B, patch_size[0], patch_size[1], C).permute(0, 3, 1, 2)


def get_cam_rays_from_bundle(
    ray_bundle: RayBundle, patch_size: Tuple[int, int]
) -> Tuple[Tensor, Tensor]:
    is_cam = is_cam_ray(ray_bundle)
    origins = unfold_ray_vec(ray_bundle.origins[is_cam], patch_size)
    directions = unfold_ray_vec(ray_bundle.directions[is_cam], patch_size)

    return origins, directions


def upsample_rays(rays: Tensor, upsample_factor: int) -> Tensor:
    assert len(rays.shape) == 4

    rays = torch.nn.functional.interpolate(
        rays, scale_factor=upsample_factor, mode="bilinear", align_corners=False
    )

    return rays


def get_ray_signal_from_ray_bundle(
    ray_bundle: RayBundle,
    patch_size: Tuple[int, int],
    upsample_factor: int = 4,
) -> Tensor:
    origins, directions = get_cam_rays_from_bundle(
        ray_bundle,
        patch_size,
    )

    origins = upsample_rays(origins, upsample_factor)
    directions = upsample_rays(directions, upsample_factor)

    rays = torch.cat([origins, directions], dim=1)
    return rays


def get_diffusion_strength(
    step: int,
    start_strength: float,
    noise_drop_start_step: int,
    noise_drop_end_step: int,
    num_inference_steps: int,
) -> float:
    min_noise = 1 / num_inference_steps + 1e-10

    if step < noise_drop_start_step:
        return start_strength

    if step >= noise_drop_end_step:
        return min_noise

    scalar = (step - noise_drop_start_step) / (
        noise_drop_end_step - noise_drop_start_step
    )

    # There will be at least one step to avoid issues with diffusion

    return max((1 - scalar) * start_strength, min_noise)


def get_pose_augmentation(
    step: int,
    augment_max_strength: PoseConfig,
    augment_direction: PoseConfig,
    max_aug_step: int,
    event: Optional[Tensor] = None,
    augment_strategy: str = "partial_linear",
    rng: Optional[torch.Generator] = None,
) -> Tensor:
    if rng is None:
        rng = torch.default_generator

    if event is None:
        event = torch.ones(6, dtype=torch.bool)

    max_strength = augment_max_strength.tensor_rad * augment_direction.tensor_deg

    if augment_strategy == "partial_linear" and step < max_aug_step:
        # Linear ramp starting at 0
        max_strength *= step / max_aug_step

    rel_aug = torch.empty_like(max_strength).uniform_(0, 1, generator=rng)

    augmentation = (
        rel_aug * max_strength * event
    )  # Uniform in [-max_strength, max_strength], zeroed out if event is False

    return augmentation
