import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import MaskedLMOutput   
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
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
            # if self.use_fast_path or inference_params is not None:
            #     return original_forward(hidden_states, inference_params)

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
            y = selective_scan_fn(
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



class BiMambaForMaskedLM(PreTrainedModel):
    config_class    = AutoConfig
    base_model_prefix = "bimamba"

    def __init__(self, config):
        super().__init__(config)                    # <-- HF init
        mamba_cfg = convert_hf_config_to_mamba(config)

        # your embedding + two Mamba directions + proj
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mamba_forward   = MambaLMHeadModel(mamba_cfg)
        self.mamba_backward  = MambaLMHeadModel(mamba_cfg)
        self.lm_head_proj    = nn.Linear(config.d_model * 2, config.d_model, bias=False)

        # your custom patches (circular conv, accept_embeds…)
        patch_mixer_forward_to_accept_embeddings(self.mamba_forward)
        patch_mixer_forward_to_accept_embeddings(self.mamba_backward)
        inject_circular_convolution(self.mamba_forward)
        inject_circular_convolution(self.mamba_backward)

        # self.post_init()  # wires up HF weight-tying & save/load

    #### Added:
    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, new_emb):
        self.token_embedding = new_emb

    def get_output_embeddings(self):
        return self.lm_head_proj

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        for backbone in (self.mamba_forward.backbone,
                         self.mamba_backward.backbone):
            for block in backbone.layers:
                block.gradient_checkpointing = True

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
    ):
        # exactly your custom logic…
        if inputs_embeds is None:
            input_ids = input_ids.long()
            inputs_embeds = self.token_embedding(input_ids)

        hid_fwd = self.mamba_forward.backbone(inputs_embeds=inputs_embeds)
        rev_emb = torch.flip(inputs_embeds, dims=[1])
        hid_bwd = self.mamba_backward.backbone(inputs_embeds=rev_emb)
        hid_bwd = torch.flip(hid_bwd, dims=[1])

        combined = torch.cat([hid_fwd, hid_bwd], dim=-1)
        projected = self.lm_head_proj(combined)
        logits    = F.linear(projected, self.token_embedding.weight)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss    = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            out = (logits, combined)
            return (loss,) + out if loss is not None else out

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=combined,
        )
