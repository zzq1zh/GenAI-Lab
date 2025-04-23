import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from einops import rearrange

def convert_hf_config_to_mamba(hf_config) -> MambaConfig:
  return MambaConfig(
      d_model=hf_config.d_model,
      d_intermediate=getattr(hf_config, "intermediate_size", 4 * hf_config.d_model),
      n_layer=getattr(hf_config, "n_layer", getattr(hf_config, "num_hidden_layers", 12)),
      vocab_size=hf_config.vocab_size,
      ssm_cfg=getattr(hf_config, "ssm_cfg", {}),
      attn_layer_idx=getattr(hf_config, "attn_layer_idx", []),
      attn_cfg=getattr(hf_config, "attn_cfg", {}),
      rms_norm=getattr(hf_config, "rms_norm", True),
      residual_in_fp32=getattr(hf_config, "residual_in_fp32", True),
      fused_add_norm=getattr(hf_config, "fused_add_norm", False),
      pad_vocab_size_multiple=getattr(hf_config, "pad_vocab_size_multiple", 8),
      tie_embeddings=getattr(hf_config, "tie_embeddings", False),
  )

def patch_mixer_forward_to_accept_embeddings(model):
    """
    Injects a new forward method into a MixerModel instance,
    allowing it to accept either input_ids or inputs_embeds.
    """

    def new_forward(self, input_ids=None, inputs_embeds=None, inference_params=None, **mixer_kwargs):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embedding(input_ids)
        else:
            raise ValueError("You must provide either input_ids or inputs_embeds.")

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        return hidden_states

    # Bind the new forward method to the instance
    model.backbone.forward = new_forward.__get__(model.backbone, model.backbone.__class__)

def inject_circular_convolution(mamba_model: torch.nn.Module):
    """
    This function injects circular convolution into the Mamba layers of a MambaLMHeadModel.
    It replaces the default causal convolution with circular padding.
    """

    def patch_mamba_layer(mamba_layer: Mamba):
        original_forward = mamba_layer.forward

        def circular_forward(self, hidden_states, inference_params=None):
            """
            Replaces the causal convolution in the Mamba layer with circular convolution.
            All other logic remains unchanged. This only affects the non-fast-path forward.
            """
            if self.use_fast_path or inference_params is not None:
                return original_forward(hidden_states, inference_params)

            batch, seqlen, dim = hidden_states.shape

            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

            x, z = xz.chunk(2, dim=1)

            # Apply circular padding before convolution
            x = F.pad(x, (self.d_conv - 1, 0), mode="circular")
            x = self.act(self.conv1d(x)[..., :seqlen])

            # Continue with the rest of the Mamba state-space computation
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) d -> b d l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) d -> b d l", l=seqlen).contiguous()

            A = -torch.exp(self.A_log.float())
            y = self.selective_scan(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = rearrange(y, "b d l -> b l d")
            return self.out_proj(y)

        # Patch the forward function of the Mamba instance
        mamba_layer.forward = circular_forward.__get__(mamba_layer, Mamba)

    # Iterate through Mamba blocks in the backbone model and apply patch
    for block in mamba_model.backbone.layers:
        patch_mamba_layer(block.mixer)

class BiMambaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mamba_config = convert_hf_config_to_mamba(config)
        print(mamba_config)

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mamba_forward = MambaLMHeadModel(mamba_config)
        self.mamba_backward = MambaLMHeadModel(mamba_config)

        # Tied embedding logic
        self.lm_head_proj = nn.Linear(config.d_model * 2, config.d_model, bias=False)

        # Inject circular convolution kernels into both directions
        patch_mixer_forward_to_accept_embeddings(self.mamba_forward)
        patch_mixer_forward_to_accept_embeddings(self.mamba_backward)
        inject_circular_convolution(self.mamba_forward)
        inject_circular_convolution(self.mamba_backward)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embed input
        input_ids = input_ids.long()
        inputs_embeds = self.token_embedding(input_ids)

        # Forward hidden
        hidden_forward = self.mamba_forward.backbone(inputs_embeds=inputs_embeds)

        # Backward hidden
        reversed_embeds = torch.flip(inputs_embeds, dims=[1])
        hidden_backward = self.mamba_backward.backbone(inputs_embeds=reversed_embeds)
        hidden_backward = torch.flip(hidden_backward, dims=[1])

        # Concat hidden
        combined_output = torch.cat([hidden_forward, hidden_backward], dim=-1)

        # Project to vocab logits via tied weight
        projected = self.lm_head_proj(combined_output)
        logits = F.linear(projected, self.token_embedding.weight)

        # Optional loss
        loss = None
        if labels is not None:
            pad_token_id = -100
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": combined_output
        }