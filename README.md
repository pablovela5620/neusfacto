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