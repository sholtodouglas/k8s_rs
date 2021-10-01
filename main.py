from flask import Flask, request



app = Flask(__name__)

# Imports
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.append('./CLIP_JAX')
sys.path.append('./jax-guided-diffusion')

import math
import io
import sys
import time
import os
import functools
import random
import time

from PIL import Image
import requests

import numpy as np
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import PRNG, Context, ParamState, Module

from tqdm.notebook import tqdm
from google.cloud import storage
# from google.colab import files

import clip_jax

from lib.script_util import model_and_diffusion_defaults, create_model, create_model_and_diffusion


class MakeCutouts(object):
    """Cutout sizes must be fixed ahead of time to avoid
    jit-compilation issues with variable sized tensors.  We
    compute them deterministically from the parameters, so that
    jit recompilation is only required when the parameters are
    changed."""

    def __init__(self, cut_size, cutn, cut_pow=1., img_size=256):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.img_size = img_size

        sideX = img_size
        sideY = img_size
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, cut_size)
        cut_us = jax.random.uniform(jax.random.PRNGKey(0), shape=[cutn])**cut_pow
        self.cutout_sizes = (min_size + cut_us * (max_size - min_size)).astype(jnp.int32).tolist()

    def key(self):
        return (self.cut_size,self.cutn,self.cut_pow,self.img_size)
    def __hash__(self):
        return hash(self.key())
    def __eq__(self, other):
        if type(other) is MakeCutouts:
            return self.key() == other.key()
        return NotImplemented

    def __call__(self, input, key):
        [b, c, h, w] = input.shape
        rng = PRNG(key)
        cutouts = []
        for (i, size) in enumerate(self.cutout_sizes):
            offsetx = jax.random.randint(rng.split(), [], 0, w - size + 1)
            offsety = jax.random.randint(rng.split(), [], 0, h - size + 1)
            cutout = jax.lax.dynamic_slice(input,
                                           [0, 0, offsety, offsetx],
                                           [b, c, size, size])
            cutout = jax.image.resize(cutout,
                                      (input.shape[0], input.shape[1],
                                       self.cut_size, self.cut_size),
                                      method='bilinear')
            cutouts.append(cutout)
        return jnp.concatenate(cutouts, axis=0)


def Normalize(mean, std):
    mean = jnp.array(mean).reshape(3,1,1)
    std = jnp.array(std).reshape(3,1,1)
    def forward(image):
        return (image - mean) / std
    return forward

def norm1(x):
    """Normalize to the unit sphere."""
    return x / x.square().sum(axis=-1, keepdims=True).sqrt()

def spherical_dist_loss(x, y):
    x = norm1(x)
    y = norm1(y)
    return (x - y).square().sum(axis=-1).sqrt().div(2).arcsin().square().mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    # input = jnp.pad(input, ((0,0), (0,0), (0,1), (0,1)), mode='edge')
    # x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    # y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    # return (x_diff**2 + y_diff**2).mean([1, 2, 3])
    x_diff = input[..., :, 1:] - input[..., :, :-1]
    y_diff = input[..., 1:, :] - input[..., :-1, :]
    return x_diff.square().mean([1,2,3]) + y_diff.square().mean([1,2,3])


model_config = model_and_diffusion_defaults()
model_config.update({
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
})


# # Load models
# print('Creating models')
# model, diffusion = create_model_and_diffusion(**model_config)
# model_params = ParamState(model.labeled_parameters_())
# model_params.initialize(jax.random.PRNGKey(0))

# print('Loading state dict...') 
# with open('256x256_diffusion_uncond.cbor', 'rb') as fp:
#     jax_state_dict = jaxtorch.cbor.load(fp)

# model.load_state_dict(model_params, jax_state_dict)


# # model_params_f16 = ParamState(model.labeled_parameters_())
# # for p in model.parameters():
# #   if model_params[p].dtype == jnp.float32:
# #     model_params_f16[p] = model_params[p].astype(jnp.bfloat16)
# #   else:
# #     model_params_f16[p] = model_params[p]

# print('Loading CLIP model...')
# image_fn, text_fn, clip_params, _ = clip_jax.load(os.getcwd()+'/local_ViT-B-32.pt')
# clip_size = 224
# normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                       std=[0.26862954, 0.26130258, 0.27577711])




def exec_model(model_params, x, timesteps, y=None):
    cx = Context(model_params, jax.random.PRNGKey(0))
    return model(cx, x, timesteps, y)
exec_model_jit = functools.partial(jax.jit(exec_model), model_params)

def cond_loss(x, t, text_embed, cur_t, key, model_params, clip_params, clip_guidance_scale, tv_scale, make_cutouts):
    n = x.shape[0]
    my_t = jnp.ones([n], dtype=jnp.int32) * cur_t
    out = diffusion.p_mean_variance(functools.partial(exec_model,model_params),
                                    x, my_t,
                                    clip_denoised=False,
                                    model_kwargs={'y': None})
    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
    x_in = out['pred_xstart'] * fac + x * (1 - fac)
    clip_in = normalize(make_cutouts(x_in.add(1).div(2), key))
    # Average the clip embeds, then compute great circle distance
    image_embeds = emb_image(clip_in, clip_params).reshape([cutn, n, 512])
    losses = spherical_dist_loss(image_embeds.mean(0), text_embed)
    tv_losses = tv_loss(x_in)
    loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale
    return -loss
base_cond_fn = jax.jit(jax.grad(cond_loss), static_argnames='make_cutouts')

def txt(prompt):
  """Returns normalized embedding."""
  text = clip_jax.tokenize([prompt])
  text_embed = text_fn(clip_params, text)
  return norm1(text_embed.reshape(512))

def emb_image(image, clip_params=None):
    return norm1(image_fn(clip_params, image))



batch_size = 1
clip_guidance_scale = 2000
tv_scale = 150
cutn = 16
cut_pow = 1.0
n_batches = 1
init_image = None
skip_timesteps = 0


def save_to_bucket(img_name, wallet_id):
    client = storage.Client()
    bucket = client.get_bucket('reality_stone_image_store')
    blob = bucket.blob(f"{wallet_id}/{img_name}")
    blob.upload_from_filename(filename=img_name)
    # remove it once done to avoid memory bloat
    os.remove(img_name)

# # prompt, seed
# def run(prompt, seed, wallet_id):
#     rng = PRNG(jax.random.PRNGKey(seed))

#     text_embed = txt(prompt)

#     init = None
#     # if init_image is not None:
#     #     init = Image.open(fetch(init_image)).convert('RGB')
#     #     init = init.resize((model_config['image_size'], model_config['image_size']), Image.LANCZOS)
#     #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

#     cur_t = None

#     make_cutouts = MakeCutouts(clip_size, cutn, cut_pow=cut_pow, img_size=model_config['image_size'])

#     def cond_fn(x, t):
#         # Triggers recompilation if cutout parameters have changed (cutn or cut_pow).
#         return base_cond_fn(x, jnp.array(t),
#                             text_embed=text_embed,
#                             cur_t=jnp.array(cur_t),
#                             key=rng.split(),
#                             model_params=model_params,
#                             clip_params=clip_params,
#                             clip_guidance_scale = clip_guidance_scale,
#                             tv_scale = tv_scale,
#                             make_cutouts=make_cutouts)

#     for i in range(n_batches):
#         cur_t = diffusion.num_timesteps - skip_timesteps - 1

#         samples = diffusion.p_sample_loop_progressive(
#             exec_model_jit,
#             (batch_size, 3, model_config['image_size'], model_config['image_size']),
#             rng=rng,
#             clip_denoised=False,
#             model_kwargs={},
#             cond_fn=cond_fn,
#             progress=tqdm,
#             skip_timesteps=skip_timesteps,
#             init_image=init,
#         )

#         for j, sample in enumerate(samples):
#             cur_t -= 1
#             if j % 10 == 0 or cur_t == -1:
#                 for k, image in enumerate(sample['pred_xstart']):
#                     filename = f'{prompt}_$$_{seed}.png'
#                     # print(k, type(image).mro())
#                     # For some reason this comes out as a numpy array. Huh?
#                     image = jnp.array(image)
#                     image = image.add(1).div(2).clamp(0, 1)
#                     image = jnp.transpose(image, (1, 2, 0))
#                     image = (image * 256).clamp(0, 255)
#                     image = np.array(image).astype('uint8')
#                     image = Image.fromarray(image)
#                     image.save(filename)
#                     save_to_bucket(filename, wallet_id)

#                     # image.save(filename) # TODO save to a bucket
#                     tqdm.write(f'Batch {i}, step {j}, output {k}:')
                    




@app.route('/post', methods=['POST'])
def post_route():
    ''''
    Expecting json of form 

    {
        prompt:
        wallet_id:
        debug: 

    }
    '''
    data = request.get_json() # should have prompt, user fields

    if request.method == 'POST':
        if data['debug'].lower() == 'true':
            print(data)
            # test_img_path = file_path+'/test.png'

            save_to_bucket('test.png', data['wallet_id'])
            time.sleep(5)
            import torch
            return f"{data['prompt']}_{str(jax.devices())}_{torch.cuda.is_available()}"
        else:
            print('Request Received: "{data}"'.format(data=data))
            run(prompt=data['prompt'], seed=random.randint(0,100), wallet_id=data['wallet_id'])
            return "Request Processed.\n"

app.run(host='0.0.0.0', debug=False, port=8000)


# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"username":"xyz","password":"xyz"}' \
#    http://0.0.0.0:8000/post
