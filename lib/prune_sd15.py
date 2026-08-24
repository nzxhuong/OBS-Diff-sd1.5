"""
lib/prune_sd15.py

SD1.5 / Hyper-SD adaptation of prune_OBS_Diff_Structured from lib/prune.py.

Changes vs. the original MMDiT version:
  1. Uses lib.unet_blocks (build_block_registry / build_target_pruned_modules)
     instead of flat `pipe.transformer.transformer_blocks[i]` indexing.
  2. attn1 (self-attn) and attn2 (cross-attn) are pruned independently with
     the plain (non-joint) OBS_Diff_Structured class -- there's no MMDiT-style
     shared text stream on SD1.5, so OBS_Diff_Structured_Joint_Attn is unused.
  3. headsize is read per-block from attn.heads / to_q.out_features instead
     of hard-coded 64, since head_dim differs across down/mid/up stages.
  4. Parallel grouping rules are rewritten for SD1.5 module names (used only
     to batch calibration forward passes -- doesn't affect correctness).

Requires lib/unet_blocks.py from the earlier step.
"""

import torch
import torch_pruning as tp
from collections import defaultdict

from .unet_blocks import build_block_registry, build_target_pruned_modules, find_layers
from .OBS_Diff_Structured import OBS_Diff_Structured
from .dataloader import get_loaders

# reuse the hook machinery unchanged -- it's architecture-agnostic
from .prune import create_hook_fn, step_info, callback_on_step_end, get_module_by_name


# ---------------------------------------------------------------------------
# 1. SD1.5 parallel-grouping rules (batches calibration forward passes;
#    purely an efficiency knob, not required for correctness)
# ---------------------------------------------------------------------------
SD15_PARALLEL_SETS = [
    {"attn1.to_q", "attn1.to_k", "attn1.to_v"},
    {"attn2.to_q", "attn2.to_k", "attn2.to_v"},
]
# Note: to_out.0 (attn1/attn2) and ff.net.2 are each their own singleton
# group below -- they're the Hessian-tracked layers for structured pruning,
# same role "attn.to_out.0" / "ff.net.2" played in the MMDiT version.


def group_modules_with_parallelism_sd15(target_pruned_modules, num_groups):
    """Same shape/contract as the original group_modules_with_parallelism,
    but keyed by string block_key instead of int block_idx, and using
    SD15_PARALLEL_SETS instead of the MMDiT joint-stream rules."""
    modules_by_block = defaultdict(list)
    for block_key, name in target_pruned_modules:
        modules_by_block[block_key].append(name)

    groupable_items = []
    for block_key in modules_by_block:
        block_modules = set(modules_by_block[block_key])
        processed = set()

        for p_set in SD15_PARALLEL_SETS:
            intersection = block_modules.intersection(p_set)
            if intersection:
                unit = [(block_key, name) for name in sorted(intersection)]
                groupable_items.append(unit)
                processed.update(intersection)

        remaining = block_modules - processed
        for name in sorted(remaining):
            groupable_items.append([(block_key, name)])

    num_items = len(groupable_items)
    if num_items == 0:
        return []

    group_size = num_items // num_groups
    remainder = num_items % num_groups
    if group_size == 0:
        group_size = 1
        num_groups = num_items
        remainder = 0

    final_groups = []
    start = 0
    for i in range(num_groups):
        end = start + group_size + (1 if i < remainder else 0)
        chunk = groupable_items[start:end]
        final_groups.append([m for unit in chunk for m in unit])
        start = end

    return final_groups


# ---------------------------------------------------------------------------
# 2. head_dim lookup -- replaces the hard-coded headsize=64
# ---------------------------------------------------------------------------
def get_attn_head_dim(basic_block, attn_name: str) -> int:
    """attn_name is 'attn1' or 'attn2'. Reads heads/out_features straight off
    the live module so it's correct per down/mid/up stage without guessing."""
    attn_module = get_module_by_name(basic_block, attn_name)
    out_features = attn_module.to_q.out_features
    heads = attn_module.heads
    assert out_features % heads == 0, (
        f"{attn_name}.to_q.out_features={out_features} not divisible by heads={heads}"
    )
    return out_features // heads


# ---------------------------------------------------------------------------
# 3. Main driver
# ---------------------------------------------------------------------------
@torch.no_grad()
def prune_OBS_Diff_Structured_SD15(args, pipe, dev, timestep_weight=None):
    """
    target_modules for structured pruning on SD1.5 are the three
    Hessian-tracked layers: attn1.to_out.0, attn2.to_out.0, ff.net.2.
    The paired layers (to_q/to_k/to_v, ff.net.0.proj) are pruned by
    dependency propagation from the indices these three return -- they are
    never hooked or Hessian-tracked directly.
    """
    print('Starting SD1.5 structured pruning...')
    dataloader = get_loaders(args.dataset, num_samples=args.num_samples)

    unet = pipe.unet
    registry = build_block_registry(unet)

    target_modules = ["attn1.to_out.0", "attn2.to_out.0", "ff.net.2"]
    target_pruned_modules = build_target_pruned_modules(
        unet, target_modules, args.minlayer, args.maxlayer
    )

    print(f"registry has {len(registry)} blocks, "
          f"{len(target_pruned_modules)} target modules found")

    modules_groups = group_modules_with_parallelism_sd15(
        target_pruned_modules, args.num_pruned_groups
    )
    print(f"divided into {len(modules_groups)} groups")
    for g_idx, group in enumerate(modules_groups):
        print(f"Group {g_idx + 1}: {group}")

    for g_idx, group_modules in enumerate(modules_groups):
        print(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")

        pruner_dict = {}
        hooks = []
        for block_key, module_name in group_modules:
            block_module = registry[block_key]
            all_module_dict = find_layers(block_module)
            module = all_module_dict[module_name]

            # No joint-attn class needed -- attn1/attn2/ff.net.2 all use the
            # plain per-layer OBS_Diff_Structured independently.
            pruner_dict[(block_key, module_name)] = OBS_Diff_Structured(module, block_key, args)
            hook_fn = create_hook_fn(block_key, module_name, pruner_dict, timestep_weight)
            hooks.append(module.register_forward_hook(hook_fn))

        print(f"Running diffusion for group {g_idx + 1} to collect activations...")
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
            print(f"  Prompts {i}: {prompts}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=torch.Generator("cuda").manual_seed(args.seed),
            )

        for hook in hooks:
            hook.remove()

        print(f"Pruning group {g_idx + 1}...")
        for block_key, module_name in group_modules:
            block_module = registry[block_key]
            sparsity = (
                args.sparsity_ratio[block_key]
                if isinstance(args.sparsity_ratio, dict)
                else args.sparsity_ratio
            )

            if module_name == "ff.net.2":
                pruner = pruner_dict[(block_key, module_name)]
                idx = pruner.struct_prune(sparsity=sparsity, percdamp=args.percdamp)
                inner_dim = pruner.columns  # ff.net.2's original in_features

                target_layer = get_module_by_name(block_module, "ff.net.2")
                target_layer_in = get_module_by_name(block_module, "ff.net.0.proj")
                idx = idx.tolist()

                # GEGLU: ff.net.0.proj outputs concat([hidden, gate]), each of
                # size inner_dim (out_features == 2*inner_dim), then does
                # hidden * gelu(gate) via chunk(2). An inner_dim-space pruned
                # index i must remove BOTH proj output channel i (hidden half)
                # and i + inner_dim (gate half), or the chunk split misaligns
                # after pruning. Non-gated activations (e.g. gelu-approximate,
                # used by SD3's FeedForward) have proj.out_features == inner_dim
                # and don't need this doubling.
                proj_out_features = target_layer_in.out_features
                if proj_out_features == 2 * inner_dim:
                    full_idx = idx + [i + inner_dim for i in idx]
                else:
                    full_idx = idx

                tp.prune_linear_in_channels(target_layer, idx)
                tp.prune_linear_out_channels(target_layer_in, full_idx)

            else:  # "attn1.to_out.0" or "attn2.to_out.0"
                attn_name = module_name.split(".")[0]  # "attn1" or "attn2"
                head_dim = get_attn_head_dim(block_module, attn_name)

                idx = pruner_dict[(block_key, module_name)].struct_prune(
                    sparsity=sparsity, percdamp=args.percdamp, headsize=head_dim
                )
                idx = idx.tolist()

                to_out = get_module_by_name(block_module, f"{attn_name}.to_out.0")
                to_q = get_module_by_name(block_module, f"{attn_name}.to_q")
                to_k = get_module_by_name(block_module, f"{attn_name}.to_k")
                to_v = get_module_by_name(block_module, f"{attn_name}.to_v")

                tp.prune_linear_in_channels(to_out, idx)
                tp.prune_linear_out_channels(to_q, idx)
                tp.prune_linear_out_channels(to_k, idx)
                tp.prune_linear_out_channels(to_v, idx)

                attn_module = get_module_by_name(block_module, attn_name)
                prev_heads = attn_module.heads
                attn_module.heads -= len(idx) // head_dim
                print(f"[{block_key}.{attn_name}] heads {prev_heads} -> {attn_module.heads}")

            pruner_dict[(block_key, module_name)].free()

        torch.cuda.empty_cache()
        print(f"Group {g_idx + 1} pruning completed.")