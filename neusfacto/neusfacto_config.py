"""neusfacto/neusfacto_config.py"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.engine.schedulers import MultiStepSchedulerConfig

from neusfacto.neusfacto import NeuSFactoModelConfig
from neusfacto.data.sdf_dataset import SDFDataset

clean_neusfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="clean-neus-facto",
        steps_per_eval_batch=5000,
        steps_per_eval_image=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=40001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[SDFDataset],
                dataparser=SDFStudioDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=1024,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=NeuSFactoModelConfig(
                overwrite_near_far_plane=True,
                near_plane=0.05,
                far_plane=2.0,
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.8,
                    beta_init=0.3,
                    use_appearance_embedding=False,
                    inside_outside=False,  # True for rooms, false for objects
                ),
                background_model="none",
                eval_num_rays_per_chunk=1024,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(max_steps=20001),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
                ),
            },
            "field_background": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": CosineDecaySchedulerConfig(
                    warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Neusfacto testing",
)
