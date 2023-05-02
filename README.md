# NeuSFacto Example
Example of registering custom models with nerfstudio
## Installation
Nerfstudio Dependancies. Follow instructions [here](https://docs.nerf.studio/en/latest/quickstart/installation.html#dependencies)
Then run

```
pip install --upgrade pip setuptools
pip install -e .
```

## Dataset generation
Given a colmap dataset processed by ns-process-data run the following to get into sdfstudio format for neusfacto
Make sure to have the omnidata repo installed https://github.com/EPFL-VILAB/omnidata
```
python nerfstudio_to_sdfstudio.py --data PATH TO DATA --output-dir PATH TO OUTPUT --data-type colmap --scene-type indoor --mono-prior --omnidata-path PATH TO OMNIDATA REPO --pretrained-models PATH-TO-PRETRAINED OMNIDATA MODELS
```

## Example command
```
ns-train neus-facto-custom --pipeline.model.sdf-field.inside-outside True --pipeline.model.sdf_field.bias 0.8 --pipeline.model.sdf_field.beta_init 0.3 --pipeline.model.mono-depth-loss-mult 0.1 --pipeline.model.mono-normal-loss-mult 0.05 --vis wandb --experiment-name corazon_sdf sdfstudio-data --data data/corazon_studio_sdf/ --include_mono_prior True
```