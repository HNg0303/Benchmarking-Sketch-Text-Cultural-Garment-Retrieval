import torch
from clip.model import CLIP


def encode_with_pseudo_tokens(clip_model: CLIP, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
    cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x
# def encode_with_pseudo_tokens(clip_model, text, pseudo_tokens, num_tokens=1):
#     # text: [batch_size, 77]
#     # pseudo_tokens: [batch_size, 768] (or [batch_size, num_tokens, 768])
    
#     x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, 77, 512/768/1024]

#     # Find where the placeholder token (259) is in the sequence
#     # nonzero() returns [index_in_batch, index_in_sequence]
#     placeholder_indices = (text == 259).nonzero(as_tuple=False)
    
#     if placeholder_indices.shape[0] == 0:
#         raise ValueError("Token 259 ($) not found in input text. Check your prompt or tokenizer.")

#     # Get the row and column indices
#     rows = placeholder_indices[:, 0]
#     cols = placeholder_indices[:, 1]

#     # Handle pseudo_tokens shape
#     # We need pseudo_tokens to be [N * num_tokens, d_model] to match the indexing
#     if pseudo_tokens.shape[0] == x.shape[0]:
#         # Batch-specific tokens
#         source_data = pseudo_tokens.reshape(-1, x.shape[-1])
#     else:
#         # One token applied to the whole batch
#         source_data = pseudo_tokens.repeat(x.shape[0], 1, 1).reshape(-1, x.shape[-1])

#     # The Flattened Assignment: This avoids the "indexing result shape" error
#     x[rows, cols] = source_data.to(x.dtype)

#     # Standard CLIP Transformer forward pass
#     x = x + clip_model.positional_embedding.type(clip_model.dtype)
#     x = x.permute(1, 0, 2)  # NLD -> LND
#     x = clip_model.transformer(x)
#     x = x.permute(1, 0, 2)  # LND -> NLD
#     x = clip_model.ln_final(x).type(clip_model.dtype)

#     # Take features from the EOT (End of Text) token
#     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
#     return x
