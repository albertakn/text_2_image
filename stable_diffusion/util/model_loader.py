from ..modules.clip import CLIP
from ..modules.encoder import VAE_Encoder
from ..modules.decoder import VAE_Decoder
from ..modules.diffusion import Diffusion
from .converter import get_model_weight_dict

def load_models_from_standard_weights(path, device):
    state_dict = get_model_weight_dict(path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }