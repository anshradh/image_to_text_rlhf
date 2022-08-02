#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiffusionPipeline, LDMTextToImagePipeline
from einops import rearrange
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
import tqdm

#%%
model_id = "CompVis/ldm-text2im-large-256"


class LDMTextToImagePipelineWithValueHead(nn.Module):
    def __init__(
        self,
        model_id,
        ff_dim,
        ff_dim_multiplier=4,
    ):
        super().__init__()
        pipeline = DiffusionPipeline.from_pretrained(model_id)
        self.bert = pipeline.bert
        self.scheduler = pipeline.scheduler
        self.tokenizer = pipeline.tokenizer
        self.unet = pipeline.unet
        self.vqvae = pipeline.vqvae

        self.v_head = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(ff_dim, ff_dim * ff_dim_multiplier),
            nn.SiLU(),
            nn.Linear(ff_dim * ff_dim_multiplier, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, 1),
        )

    def forward(
        self,
        prompt,
        generator=None,
        torch_device=None,
        eta=0.0,
        guidance_scale=1.0,
        num_inference_steps=50,
    ):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = len(prompt)

        self.unet.to(torch_device)
        self.vqvae.to(torch_device)
        self.bert.to(torch_device)

        # get unconditional embeddings for classifier free guidance
        if guidance_scale != 1.0:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            uncond_embeddings = self.bert(uncond_input.input_ids.to(torch_device))

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt"
        )
        text_embeddings = self.bert(text_input.input_ids.to(torch_device))

        latents = torch.randn(
            (
                batch_size,
                self.unet.in_channels,
                self.unet.sample_size,
                self.unet.sample_size,
            ),
            generator=generator,
        )
        latents = latents.to(torch_device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            print(t)
            if guidance_scale == 1.0:
                # guidance_scale of 1 means no guidance
                latents_input = latents
                context = text_embeddings
            else:
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                latents_input = torch.cat([latents] * 2)
                context = torch.cat([uncond_embeddings, text_embeddings])

            # predict the noise residual
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)[
                "sample"
            ]
            # perform guidance
            if guidance_scale != 1.0:
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_prediction_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, eta)["prev_sample"]

        # scale image latents
        latents = 1 / 0.18215 * latents

        image = self.vqvae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        values = self.v_head(image)

        return image, values


test = LDMTextToImagePipelineWithValueHead(model_id, 28, 4)
test(prompt="a white dog runs")
#%%
# %%
