import itertools
import math
import random
import os

import numpy as np
import torch
import torch.nn.functional as F
import accelerate

from accelerate import Accelerator
from tqdm.auto import tqdm
from .utils import get_time_embedding, resize_clip_embedding, freeze_params
from .dataset import TextualInversionDataset, create_dataloader
from ..stable_diffusion.pipeline.ddpm import DDPMSampler


def save_progress(text_encoder, placeholder_token, placeholder_token_id, accelerator, save_path):
    # logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).embedding.token_embedding.weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)
    

def training_function(tokenizer, models, train_dataset, hyperparameters):
    placeholder_token_id = hyperparameters["placeholder_token_id"]
    placeholder_token = hyperparameters["placeholder_token"]
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    device = hyperparameters['device']

    accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )

    train_dataloader = create_dataloader(train_dataset, train_batch_size)

    if hyperparameters["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

    optimizer = torch.optim.AdamW(
            models['clip'].embedding.token_embedding.parameters(),  # only optimize the embeddings
            lr=learning_rate,
        )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
            models['clip'], optimizer, train_dataloader
        )

    weight_dtype = torch.float32
    # Move vae and unet to device
    models['encoder'].to(accelerator.device, dtype=weight_dtype)
    models['decoder'].to(accelerator.device, dtype=weight_dtype)
    models['diffusion'].to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    models['encoder'].eval()
    models['decoder'].eval()
    # Keep unet in train mode to enable gradient checkpointing
    models['diffusion'].train()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    WIDTH = 512
    HEIGHT = 512
    LATENTS_WIDTH = WIDTH // 8
    LATENTS_HEIGHT = HEIGHT // 8
    generator = torch.Generator(device=device)
    sampler = DDPMSampler(generator)
    generator = torch.Generator(device=device)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models['clip']):
                # Convert images to latent space
                latents_shape = (batch["pixel_values"].shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH)
                encoder_noise = torch.randn(latents_shape, generator=generator, device=device).to(dtype=weight_dtype)
                latents = models['encoder'](batch["pixel_values"].to(dtype=weight_dtype), encoder_noise).detach()

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, sampler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents, noise = sampler.add_noise(latents, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = models['clip'](batch["input_ids"])[0]

                # Predict the noise residual
                time_embedding = get_time_embedding(timesteps).to(device)
                noise_pred = models['diffusion'](noisy_latents, encoder_hidden_states.to(weight_dtype), time_embedding.to(weight_dtype))

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = models['clip'].module.embedding.token_embedding.weight.grad
                else:
                    grads = models['clip'].embedding.token_embedding.weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        # Also save the newly trained embeddings
        save_path = os.path.join(output_dir, f"learned_embeds.bin")
        save_progress(models['clip'], placeholder_token, placeholder_token_id, accelerator, save_path)


def run_textual_inversion(models, tokenizer, hyperparameters):

    placeholder_token = hyperparameters["placeholder_token"]
    initializer_token = hyperparameters["initializer_token"]
    what_to_teach = hyperparameters["what_to_teach"]
    save_path = hyperparameters["save_path"]

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    hyperparameters["placeholder_token_id"] = placeholder_token_id

    #We have added the placeholder_token in the tokenizer so we resize the token embeddings here,
    #this will a new embedding vector in the token embeddings for our placeholder_token

    resize_clip_embedding(clip_model=models['clip'], new_num_tokens=len(tokenizer))

    #Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = models['clip'].embedding.token_embedding.weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]


    #In Textual-Inversion we only train the newly added embedding vector, so lets freeze rest of the model parameters here
    # Freeze vae and unet
    freeze_params(models['diffusion'].parameters())
    freeze_params(models['encoder'].parameters())
    freeze_params(models['decoder'].parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        models['clip'].layers.parameters(),
        models['clip'].layernorm.parameters(),
    )
    freeze_params(params_to_freeze)

    
    train_dataset = TextualInversionDataset(
      data_root=save_path,
      tokenizer=tokenizer,
      size=512,
      placeholder_token=placeholder_token,
      repeats=100,
      learnable_property=what_to_teach, #Option selected above between object and style
      set="train",
    )

    accelerate.notebook_launcher(training_function, args=(tokenizer, models, train_dataset, hyperparameters))

    for param in itertools.chain(models['diffusion'].parameters(), models['clip'].parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()
                        
    return models
