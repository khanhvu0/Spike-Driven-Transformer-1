from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *
import json
import re


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pooling_stat="1111",
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
        auto_skip=False,
        skip_threshold=0.01,
        log_file=None,
        # firing_thresholds=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.auto_skip = auto_skip
        self.skip_threshold = skip_threshold
        self.skip_layers = []

        # Parse log file to determine which layers to skip if auto_skip is enabled
        if auto_skip and log_file:
            self.skip_layers = self._parse_log_file(log_file, skip_threshold)
            print(f"Auto-skipping layers: {self.skip_layers}")

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                    skip_attention=j in self.skip_layers,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _parse_log_file(self, log_file, threshold):
        """Parse log file to determine which layers to skip based on firing rate threshold"""
        skip_layers = []
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
            # Find the firing_rate JSON section in the log
            # The pattern to match: "firing_rate: " followed by newline, then "INFO: {" and the entire JSON
            firing_rate_start = log_content.find('firing_rate: \n')
            if firing_rate_start != -1:
                # Find the start of JSON data after "INFO: "
                json_start = log_content.find('INFO: {', firing_rate_start)
                if json_start != -1:
                    json_start += 6  # Move past "INFO: "
                    
                    # Find the end of the JSON data by finding the last curly brace
                    # before the next section or end of file
                    json_end = log_content.find('\n\n', json_start)
                    if json_end == -1:  # If no newlines, take until end of file
                        json_end = len(log_content)
                    
                    # Extract the JSON string
                    firing_rate_json = log_content[json_start:json_end].strip()
                    
                    try:
                        # Parse the JSON data
                        firing_rate_data = json.loads(firing_rate_json)
                        
                        # Debug print
                        print(f"Successfully parsed firing rate data with {len(firing_rate_data)} timesteps")
                        
                        # For each layer, calculate average firing rate across all timesteps
                        for layer in range(self.depths):
                            layer_key = f"MS_SSA_Conv{layer}_x_after_qkv"
                            
                            # Calculate average firing rate across all timesteps
                            rates_sum = 0.0
                            valid_timesteps = 0
                            rates = []
                            
                            for t in range(self.T):
                                t_key = f"t{t}"
                                if t_key in firing_rate_data and layer_key in firing_rate_data[t_key]:
                                    firing_rate = firing_rate_data[t_key][layer_key]
                                    rates.append(firing_rate)
                                    rates_sum += firing_rate
                                    valid_timesteps += 1
                            
                            # Skip if no valid data found
                            if valid_timesteps == 0:
                                continue
                                
                            avg_rate = rates_sum / valid_timesteps
                            print(f"Layer {layer} average firing rate: {avg_rate:.6f} (rates: {rates})")
                            
                            # Skip layer if average firing rate is below threshold
                            if avg_rate <= threshold:
                                skip_layers.append(layer)
                                print(f"Layer {layer} will be skipped: avg rate {avg_rate:.6f} ≤ threshold {threshold}")
                        
                        if not skip_layers:
                            print(f"No layers will be skipped (threshold = {threshold})")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in log file: {e}")
                        print(f"JSON snippet: {firing_rate_json[:200]}...")
                else:
                    print("Could not find JSON data after 'firing_rate:' in log file")
            else:
                print("Could not find 'firing_rate:' section in log file")
        except Exception as e:
            print(f"Error processing log file {log_file}: {e}")
            
        return skip_layers

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _, hook = patch_embed(x, hook=hook)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
        return x, hook

    def report_sdsa_stats(self):
        """Report SDSA skip statistics for all transformer blocks"""
        return None


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
