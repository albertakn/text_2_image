# text_2_image
This github contains the implementation of stable diffusion and several methods for fine tuning it.
You can find out more about these concepts here:
  - https://arxiv.org/abs/2112.10752
  - https://arxiv.org/abs/2208.01618
  - https://arxiv.org/abs/2208.12242

## Download weigts:
Before you start working with the model, you need to download v1-5-pruned-emaonly.ckpt
file from https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main and save it in the data directory.
So your folders should be like this:
   text_2_image(-main)/
    ├─ data/
    │  ├─ v1-5-pruned-emaonly.ckpt/
    │  ├─ ...
    ├─ stable_diffusion/
    │  ├─ modules/
    │  ├─ ...
    ├─ dreambooth/
    │  ├─ dreambooth.py
    │  ├─ ...
    ...
    └─

    text_2_image(-main)/
    ├─ data/
    │  ├─ v1-5-pruned-emaonly.ckpt
    │  ├─ ...
    ├─ stable_diffusion/
    │  ├─ modules/
    │  ├─ ...
    ├─ dreambooth/ 
    │  ├─ 
    │  ├─ 
    └── README.md

## How to use:
Example of usage can be found in the following files:
  - stable_diffusion_usage.ipynb
  - textual_inversion_usage.ipynb
  - dreambooth_usage.ipynb
Comparison of fine-tuning methods are in the comparison.ipynb

## References:
- https://github.com/CompVis/stable-diffusion/tree/main
- https://github.com/kjsman/stable-diffusion-pytorch/tree/main
- https://github.com/huggingface/diffusers/tree/2a8cf8e39fca6fb3e92b3c36e15358dbdc404de7/examples/dreambooth
- https://github.com/huggingface/diffusers/tree/2a8cf8e39fca6fb3e92b3c36e15358dbdc404de7/examples/textual_inversion
