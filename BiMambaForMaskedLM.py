import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

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

class BiMambaForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mamba_config = convert_hf_config_to_mamba(config)
        print(mamba_config)
        self.mamba_forward = MambaLMHeadModel(mamba_config)
        self.mamba_backward = MambaLMHeadModel(mamba_config)

        self.lm_head = nn.Linear(config.d_model * 2, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward hidden
        hidden_forward = self.mamba_forward.backbone(input_ids)

        # Backward hidden
        reversed_ids = torch.flip(input_ids, dims=[1])
        hidden_backward = self.mamba_backward.backbone(reversed_ids)
        hidden_backward = torch.flip(hidden_backward, dims=[1])

        # Concat hidden
        combined_output = torch.cat([hidden_forward, hidden_backward], dim=-1)

        # Project to logits
        logits = self.lm_head(combined_output)

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