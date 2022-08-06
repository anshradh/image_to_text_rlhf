#%%
DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# CLIP model
CLIP_REPO = "openai/clip-vit-large-patch14"
CLIP_COMMIT_ID = None
#%%
import jax
import jax.numpy as jnp

# check how many devices are available
jax.local_device_count()
#%%
# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# Load dalle-mini
model = DalleBart.from_pretrained(
    DALLE_MODEL,
    revision=DALLE_COMMIT_ID,
    dtype=jnp.float16,
)

# Load VQGAN
vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False)

# Load CLIP
clip = FlaxCLIPModel.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
# %%
print(model._params_shape_tree["model"])
#%%

#%%
