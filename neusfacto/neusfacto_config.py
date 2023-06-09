"""neusfacto/neusfacto_config.py"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datamanagers.sdf_datamanager import SDFDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig

from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.engine.schedulers import MultiStepSchedulerConfig

neusfacto_method = MethodSpecification(
    config=TrainerConfig(
        method_name="neus-facto-custom",
        steps_per_eval_image=1000,
        steps_per_eval_batch=1000,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=SDFDataManagerConfig(
                dataparser=SDFStudioDataParserConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=1024,
                # train_num_images_to_sample_from=1,
                # train_num_times_to_repeat_images=0,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=NeuSFactoModelConfig(
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=2,
                    num_layers_color=2,
                    hidden_dim=256,
                    bias=0.8,
                    beta_init=0.3,
                    use_appearance_embedding=False,
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
    description="NeuS Facto",
)
