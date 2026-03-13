# ============================================================
# ONNX export — convert a trained MaskablePPO policy to ONNX
#
# Exported model spec (obs_dim / action_dim inferred from model):
#   SF:  Input "obs" (batch, 96),  Output "action_logits" (batch, 24)
#   DRT: Input "obs" (batch, 157), Output "action_logits" (batch, 9)
#
# Action masking is applied by the caller before taking argmax.
# ============================================================

from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn


class _PolicyONNXWrapper(nn.Module):
    """
    Thin wrapper around SB3 MlpPolicy actor layers.

    Call chain:  obs → features_extractor → mlp_extractor → action_net
    Output is raw action logits (before softmax / masking).
    """

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


def export_to_onnx(
    checkpoint_path: pathlib.Path,
    output_path: pathlib.Path,
    opset_version: int = 17,
) -> dict:
    """
    Load a MaskablePPO checkpoint and export the actor head to ONNX.

    Args:
        checkpoint_path: SB3 .zip file produced by training_worker.py.
        output_path:     Destination .onnx file path.
        opset_version:   ONNX opset to target (default 17).

    Returns:
        Metadata dict: model_id, onnx_path, input_shape, output_shape,
        opset_version.
    """
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(checkpoint_path))
    model.policy.set_training_mode(False)

    wrapper = _PolicyONNXWrapper(model.policy)
    wrapper.eval()

    obs_dim = model.observation_space.shape[0]
    action_dim = model.action_space.n
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        str(output_path),
        input_names=["obs"],
        output_names=["action_logits"],
        opset_version=opset_version,
        dynamic_axes={
            "obs": {0: "batch"},
            "action_logits": {0: "batch"},
        },
    )

    return {
        "model_id": output_path.stem,
        "onnx_path": str(output_path),
        "input_shape": [1, obs_dim],
        "output_shape": [1, action_dim],
        "opset_version": opset_version,
    }
