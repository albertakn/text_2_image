import os
import itertools
import math
import hashlib
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from .dataset import PromptDataset, DreamBoothDataset, collate_fn
from .utils import get_time_embedding, freeze_params
from ..stable_diffusion.pipeline.ddpm import DDPMSampler


def run_dreambooth(models, tokenizer, hyperparameters):
    logging_dir = Path(hyperparameters["output_dir"], "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"]
    )

    if hyperparameters["train_text_encoder"] and hyperparameters["gradient_accumulation_steps"] > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    set_seed(hyperparameters["seed"])


    # Generate class images if prior preservation is enabled.
    if hyperparameters["with_prior_preservation"]:
        class_images_dir = Path(hyperparameters["class_data_dir"])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))


        if cur_class_images < hyperparameters["num_class_images"]:

            num_new_images = hyperparameters["num_class_images"] - cur_class_images

            for i in tqdm(
                range(num_new_images),
                desc="Generating class images for prior_preservation",
                disable=not accelerator.is_local_main_process
            ):
                image = generate(prompt=hyperparameters["class_prompt"],
                                uncond_prompt="",
                                do_cfg=True,
                                cfg_scale=8,
                                sampler_name="ddpm",
                                n_inference_steps=50,
                                models=models,
                                device="cuda",
                                idle_device="cuda",
                                tokenizer=tokenizer)

                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{i + cur_class_images}-{hash_image}.jpg"
                Image.fromarray(image).save(image_filename)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(hyperparameters['output_dir'], exist_ok=True)

    vae_encoder = models['encoder']
    vae_decoder = models['decoder']
    text_encoder = models['clip']
    unet = models['diffusion']

    freeze_params(vae_encoder.parameters())
    freeze_params(vae_decoder.parameters())

    if not hyperparameters['train_text_encoder']:
        freeze_params(text_encoder.parameters())

    if hyperparameters['scale_lr']:
        hyperparameters['learning_rate'] = (
            hyperparameters['learning_rate'] * hyperparameters['gradient_accumulation_steps'] * hyperparameters['train_batch_size'] * accelerator.num_processes
        )
    
    if hyperparameters['use_8bit_adam']:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW   

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if hyperparameters['train_text_encoder'] else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=hyperparameters['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )    

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=hyperparameters['instance_data_dir'],
        instance_prompt=hyperparameters['instance_prompt'],
        class_data_root=hyperparameters['class_data_dir'] if hyperparameters['with_prior_preservation'] else None,
        class_prompt=hyperparameters['class_prompt'],
        class_num=hyperparameters['num_class_images'],
        tokenizer=tokenizer,
        size=hyperparameters['resolution']
    ) 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters['train_batch_size'],
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, hyperparameters['with_prior_preservation'])
    )

    # Prepare everything with our `accelerator`.
    if hyperparameters['train_text_encoder']:
        unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader
        )
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
    )
        
    weight_dtype = torch.float32

    # Move vae and text_encoder to device and cast to weight_dtype

    vae_encoder.to(accelerator.device, dtype=weight_dtype)
    vae_decoder.to(accelerator.device, dtype=weight_dtype)

    if not hyperparameters['train_text_encoder']:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / hyperparameters['gradient_accumulation_steps'])
    # Afterwards we recalculate our number of training epochs
    hyperparameters['num_train_epochs'] = math.ceil(hyperparameters['max_train_steps'] / num_update_steps_per_epoch)

    # Train!
    total_batch_size = hyperparameters['train_batch_size'] * accelerator.num_processes * hyperparameters['gradient_accumulation_steps']
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
            range(0, hyperparameters['max_train_steps']),
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
    )

    WIDTH = 512
    HEIGHT = 512
    LATENTS_WIDTH = WIDTH // 8
    LATENTS_HEIGHT = HEIGHT // 8
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(hyperparameters["seed"])
    sampler = DDPMSampler(generator)


    for epoch in range(first_epoch, hyperparameters['num_train_epochs']):
        unet.train()
        if hyperparameters['train_text_encoder']:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                # Convert images to latent
                latents_shape = (batch["pixel_values"].shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH)
                encoder_noise = torch.randn(latents_shape, generator=generator, device=accelerator.device).to(dtype=weight_dtype)
                latents = models['encoder'](pixel_values, encoder_noise)


                bsz, channels = latents.shape[0], latents.shape[1]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, sampler.num_train_timesteps, (1,), device=accelerator.device).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents, noise = sampler.add_noise(latents, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = models['clip'](batch["input_ids"])[0]


                # Predict the noise residual
                time_embedding = get_time_embedding(timesteps).to(accelerator.device)
                noise_pred = unet(noisy_latents, encoder_hidden_states.to(weight_dtype), time_embedding.to(weight_dtype))


                # loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                target = noise
                if hyperparameters['with_prior_preservation']:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                # Compute instance loss
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                if hyperparameters['with_prior_preservation']:
                    # Add the prior loss to the instance loss.
                    loss = loss + hyperparameters['prior_loss_weight'] * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if hyperparameters['train_text_encoder']
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
                    if param.grad is not None:
                        del param.grad  # free some memory
                    torch.cuda.empty_cache()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= hyperparameters['max_train_steps']:
                break


    accelerator.wait_for_everyone()
    accelerator.end_training()

    models = {'diffusion':unet,
              'clip':text_encoder,
              'encoder':vae_encoder,
              'decoder':vae_decoder}
    
    return models
