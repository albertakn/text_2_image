import torch
from torch import nn


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def resize_clip_embedding(clip_model, new_num_tokens):
    token_embedding = clip_model.embedding.token_embedding
    old_num_tokens, old_embedding_dim = token_embedding.weight.shape

    # Лучше используй RaisevalueError !!!!!
    assert new_num_tokens > old_num_tokens

    # Creating new embedding layer with more entries
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

    # Setting device and type accordingly
    new_embeddings.to(token_embedding.weight.device,
                      dtype=token_embedding.weight.dtype)

    # Copying the old entries
    new_embeddings.weight.data[:old_num_tokens, :] = token_embedding.weight.data[:old_num_tokens, :]
    clip_model.embedding.token_embedding = new_embeddings
    

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
