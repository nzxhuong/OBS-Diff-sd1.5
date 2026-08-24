"""
UNet block-enumeration helper for adapting OBS-Diff's prune.py to SD1.5 / Hyper-SD.

SD3's transformer exposes a flat, indexable `pipe.transformer.transformer_blocks`
list, so the original code could just do `blocks[i]` and use an int as the
dict key everywhere. SD1.5's UNet2DConditionModel has no such flat list --
prunable modules live inside nested down_blocks / mid_block / up_blocks, and
some stages have attention (CrossAttnDownBlock2D / CrossAttnUpBlock2D /
UNetMidBlock2DCrossAttn) while others don't (plain DownBlock2D / UpBlock2D).

This module replaces the "blocks[block_idx]" pattern with:
  1. enumerate_unet_blocks(unet)  -> ordered list of BlockRef, each with a
     unique string `key` (e.g. "down_blocks.0.attentions.0.transformer_blocks.0")
  2. a registry dict {key: nn.Module} you index into instead of `blocks[i]`

Everywhere prune.py did `blocks[block_idx]` you now do `registry[block_key]`.
Everywhere it used `block_idx` as a dict/print key, a string key works fine
as a drop-in (dicts don't care whether keys are int or str).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import torch.nn as nn


@dataclass
class BlockRef:
    key: str          # unique path string, use as the dict key instead of int block_idx
    module: nn.Module # the BasicTransformerBlock or ResnetBlock2D itself
    kind: str          # "transformer_block" or "resnet"
    stage: str         # "down", "mid", "up" -- useful for reporting / filtering


def _enumerate_stage(stage_name: str, stage_blocks, prefix: str) -> List[BlockRef]:
    """Walk one down_blocks[i] / up_blocks[i] / mid_block entry."""
    refs = []
    for stage_idx, block in enumerate(stage_blocks) if isinstance(stage_blocks, list) else [(0, stage_blocks)]:
        block_prefix = f"{prefix}.{stage_idx}" if isinstance(stage_blocks, list) else prefix

        # Resnets: present in every stage type
        resnets = getattr(block, "resnets", [])
        for r_idx, resnet in enumerate(resnets):
            refs.append(BlockRef(
                key=f"{block_prefix}.resnets.{r_idx}",
                module=resnet,
                kind="resnet",
                stage=stage_name,
            ))

        # Attentions: only CrossAttn*Block2D / UNetMidBlock2DCrossAttn have these.
        # Plain DownBlock2D / UpBlock2D (e.g. the last down stage in SD1.5) don't.
        attentions = getattr(block, "attentions", None)
        if attentions is not None:
            for a_idx, transformer2d in enumerate(attentions):
                # Transformer2DModel wraps a ModuleList of BasicTransformerBlock,
                # length 1 for standard SD1.5 (SDXL uses >1 per stage).
                for t_idx, basic_block in enumerate(transformer2d.transformer_blocks):
                    refs.append(BlockRef(
                        key=f"{block_prefix}.attentions.{a_idx}.transformer_blocks.{t_idx}",
                        module=basic_block,
                        kind="transformer_block",
                        stage=stage_name,
                    ))
    return refs


def enumerate_unet_blocks(unet: nn.Module) -> List[BlockRef]:
    """
    Ordered list of every prunable container block in a diffusers UNet2DConditionModel:
    BasicTransformerBlock (has attn1/attn2/ff) and ResnetBlock2D (has conv1/conv2/
    conv_shortcut). Order follows the forward pass: down_blocks, mid_block, up_blocks.
    """
    refs: List[BlockRef] = []

    for i, down_block in enumerate(unet.down_blocks):
        refs.extend(_enumerate_stage("down", [down_block], f"down_blocks.{i}"))

    if unet.mid_block is not None:
        refs.extend(_enumerate_stage("mid", unet.mid_block, "mid_block"))

    for i, up_block in enumerate(unet.up_blocks):
        refs.extend(_enumerate_stage("up", [up_block], f"up_blocks.{i}"))

    return refs


def build_block_registry(unet: nn.Module) -> Dict[str, nn.Module]:
    """{block_key: module} -- index this instead of `blocks[block_idx]`."""
    return {ref.key: ref.module for ref in enumerate_unet_blocks(unet)}


def find_layers(module, layers=(nn.Linear, nn.Conv2d), name=''):
    """Same recursive submodule finder prune.py already has -- reused here
    unchanged, just imported instead of redefined, to scope it per block."""
    if isinstance(module, tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def build_target_pruned_modules(
    unet: nn.Module,
    target_modules: List[str],
    minlayer: Optional[int] = None,
    maxlayer: Optional[int] = None,
) -> List[tuple]:
    """
    Drop-in replacement for this block in prune.py:

        blocks = pipe.transformer.transformer_blocks
        target_pruned_modules = []
        for i in range(args.minlayer, args.maxlayer):
            block = blocks[i]
            all_module_dict = find_layers(block)
            for name in target_modules:
                if name in all_module_dict:
                    target_pruned_modules.append((i, name))

    Returns a list of (block_key: str, module_name: str) tuples. block_key is
    now a string path, not an int -- every place downstream that used
    block_idx as a dict key or print value works unchanged, since dicts don't
    care about key type. Places that did `blocks[block_idx]` need to become
    `registry[block_key]` (see build_block_registry above).

    minlayer/maxlayer slice over the *flattened enumeration order* (down ->
    mid -> up), which is the closest analog to the original int-indexed
    range() over transformer_blocks. If you'd rather filter by named stage
    ("only down_blocks", "only up_blocks.2", etc.) filter enumerate_unet_blocks()
    output directly instead of using minlayer/maxlayer.
    """
    all_refs = enumerate_unet_blocks(unet)
    if minlayer is not None or maxlayer is not None:
        lo = minlayer if minlayer is not None else 0
        hi = maxlayer if maxlayer is not None else len(all_refs)
        all_refs = all_refs[lo:hi]

    target_pruned_modules = []
    for ref in all_refs:
        all_module_dict = find_layers(ref.module)
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((ref.key, name))

    return target_pruned_modules
