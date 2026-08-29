import torch
import torch_pruning as tp
from collections import defaultdict

from .unet_blocks import build_block_registry, build_target_pruned_modules, find_layers
from .OBS_Diff import OBS_Diff
from .OBS_Diff_Structured import OBS_Diff_Structured
from .dataloader import get_loaders

from .prune import create_hook_fn, step_info, callback_on_step_end, get_module_by_name


SD15_PARALLEL_SETS = [
    {"attn1.to_q", "attn1.to_k", "attn1.to_v"},
    {"attn2.to_q", "attn2.to_k", "attn2.to_v"},
]

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

    target_modules = ["attn1.to_out.0", "attn2.to_out.0", "ff.net.2", "conv2"]
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

            pruner_dict[(block_key, module_name)] = OBS_Diff_Structured(module, block_key, args)
            hook_fn = create_hook_fn(block_key, module_name, pruner_dict, timestep_weight)
            hooks.append(module.register_forward_hook(hook_fn))

        print(f"Running diffusion for group {g_idx + 1} to collect activations...")
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_pairs = dataloader[i * batch_size:(i + 1) * batch_size]
            prompts, negatives = zip(*batch_pairs) 
            prompts, negatives = list(prompts), list(negatives)
            print(f"  Prompts {i}: {prompts}")
            print(f"  Negatives {i}: {negatives}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                negative_prompt=negatives,
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

                proj_out_features = target_layer_in.out_features
                if proj_out_features == 2 * inner_dim:
                    full_idx = idx + [i + inner_dim for i in idx]
                else:
                    full_idx = idx

                tp.prune_linear_in_channels(target_layer, idx)
                tp.prune_linear_out_channels(target_layer_in, full_idx)

            elif module_name == "conv2":
                resnet = block_module
                conv1 = get_module_by_name(resnet, "conv1")
                conv2 = get_module_by_name(resnet, "conv2")
                norm2 = get_module_by_name(resnet, "norm2")
                time_emb_proj = get_module_by_name(resnet, "time_emb_proj")

                assert resnet.time_embedding_norm == "default", (
                    f"[{block_key}] time_embedding_norm='{resnet.time_embedding_norm}' "
                    f"not supported -- 'scale_shift' chunks temb into (scale, shift) "
                    f"inside forward(), which needs separate index handling."
                )
                num_groups = norm2.num_groups
                khkw = conv2.kernel_size[0] * conv2.kernel_size[1]

                pruner = pruner_dict[(block_key, module_name)]
                idx = pruner.struct_prune(
                    sparsity=sparsity, percdamp=args.percdamp,
                    headsize=khkw, channel_align=num_groups,
                )
                channel_idx = sorted(set((idx // khkw).tolist()))

                if len(channel_idx) == 0:
                    print(f"[{block_key}] no conv2 channels pruned "
                          f"(sparsity too low for channel_align={num_groups})")
                else:
                    tp.prune_conv_out_channels(conv1, channel_idx)
                    tp.prune_groupnorm_out_channels(norm2, channel_idx)
                    tp.prune_linear_out_channels(time_emb_proj, channel_idx)
                    tp.prune_conv_in_channels(conv2, channel_idx)
                    print(f"[{block_key}] pruned {len(channel_idx)} internal "
                          f"channels (num_groups={num_groups})")

            else: 
                attn_name = module_name.split(".")[0] 
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


SD15_UNSTRUCTURED_TARGETS = [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


@torch.no_grad()
def prune_OBS_Diff_SD15(args, pipe, dev, prune_n=0, prune_m=0, timestep_weight=None):
    """
    SD1.5 adaptation of prune_OBS_Diff from lib/prune.py. Every module in
    SD15_UNSTRUCTURED_TARGETS is hooked, Hessian-tracked, and pruned directly
    via OBS_Diff.fasterprune -- no channel removal, no dependency wiring.
    """
    print('Starting SD1.5 unstructured pruning...')
    dataloader = get_loaders(args.dataset, num_samples=args.num_samples)

    unet = pipe.unet
    registry = build_block_registry(unet)

    target_pruned_modules = build_target_pruned_modules(
        unet, SD15_UNSTRUCTURED_TARGETS, args.minlayer, args.maxlayer
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

            pruner_dict[(block_key, module_name)] = OBS_Diff(module, args)
            hook_fn = create_hook_fn(block_key, module_name, pruner_dict, timestep_weight)
            hooks.append(module.register_forward_hook(hook_fn))

        print(f"Running diffusion for group {g_idx + 1} to collect activations...")
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_pairs = dataloader[i * batch_size:(i + 1) * batch_size]
            prompts, negatives = zip(*batch_pairs)  # unzip (prompt, negative) tuples
            prompts, negatives = list(prompts), list(negatives)
            print(f"  Prompts {i}: {prompts}")
            print(f"  Negatives {i}: {negatives}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                negative_prompt=negatives,
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
            sparsity = (
                args.sparsity_ratio[block_key]
                if isinstance(args.sparsity_ratio, dict)
                else args.sparsity_ratio
            )
            print(f"Pruning {block_key}.{module_name} at sparsity {sparsity}")
            pruner_dict[(block_key, module_name)].fasterprune(
                sparsity=sparsity,
                percdamp=args.percdamp,
                prunen=prune_n,
                prunem=prune_m,
            )
            pruner_dict[(block_key, module_name)].free()

        torch.cuda.empty_cache()
        print(f"Group {g_idx + 1} pruning completed.")