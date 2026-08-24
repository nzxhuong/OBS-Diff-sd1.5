import argparse
import os 
import numpy as np
import torch
from lib.prune import prune_OBS_Diff, check_sparsity, check_size
from lib.prune_sd15 import prune_OBS_Diff_Structured_SD15
from diffusers import StableDiffusionPipeline
from lib.unet_blocks import enumerate_unet_blocks
from lib.prune_sd15 import prune_OBS_Diff_Structured_SD15

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='text-to-image model, e.g. SD3')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "structured"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "OBS-Diff", "OBS-Diff-Structured", "dsnot", "magnitude_structured"])
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--dataset', type=str, default="gcc3m", help='Dataset to use for calibration.')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use for calibration.')
    parser.add_argument('--minlayer', type=int, default=None, help='Minimum layer to prune')
    parser.add_argument('--maxlayer', type=int, default=None, help='Maximum layer to prune')
    parser.add_argument('--demo_evaluate', action="store_true", help="A single image evaluation by the pruned model")
    parser.add_argument("--demo_dir", type=str, default="eval_output.png", help="Path to save the demo images.")
    parser.add_argument("--num_pruned_groups", type=int, default=4, help="Number of pruned groups.")
    parser.add_argument("--timestep_weight_strategy", type=str, default="uniform", 
                       choices=["uniform", "linear_increase", "linear_decrease", "log_increase", "log_decrease"], help="Timestep weight strategy for Hessian update")
    parser.add_argument("--timestep_min_weight", type=float, default=0.8, help="Min weight for timestep-aware weighting")
    parser.add_argument("--timestep_max_weight", type=float, default=1.2, help="Max weight for timestep-aware weighting")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--height", type=int, default=512, help="Height of the image")
    parser.add_argument("--width", type=int, default=512, help="Width of the image")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--no_compensate", action="store_true", help="Skip error compensation in OBS-Diff")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Hessian dampening factor")

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_type != "structured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
  
    device = torch.device("cuda:0")
   
    
    print(f"loading model {args.model_path}")
    pipe = StableDiffusionPipeline.from_single_file(
        args.model_path,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.unet.eval()

    # if args.minlayer is not None and args.maxlayer is not None:
    #     args.minlayer = max(args.minlayer, 0)
    #     args.maxlayer = min(args.maxlayer, pipe.transformer.config.num_layers)
    # elif args.minlayer is not None:
    #     args.minlayer = max(args.minlayer, 0)
    #     args.maxlayer = pipe.transformer.config.num_layers
    # elif args.maxlayer is not None:
    #     args.maxlayer = min(args.maxlayer, pipe.transformer.config.num_layers)
    #     args.minlayer = 0
    # else:
    #     args.minlayer = 0
    #     args.maxlayer = pipe.transformer.config.num_layers
    num_blocks = len(enumerate_unet_blocks(pipe.unet))
    print(f"UNet has {num_blocks} prunable blocks total")

    if args.minlayer is None:
        args.minlayer = 0
    if args.maxlayer is None:
        args.maxlayer = num_blocks

    # # To ensure the last layer is not pruned (we prune the complete MMDiT layers in structured pruning)
    print(f"pruning from layer {args.minlayer} to {args.maxlayer}")
    print(f"use device {device}")

   
    target_modules = ["attn1.to_out.0", "attn2.to_out.0", "ff.net.2"]

    if args.sparsity_type == "structured":

        target_modules = [
            "ff.net.2",
            "ff_context.net.2",
            "attn.to_out.0"
        ]

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "OBS-Diff":

            if args.timestep_weight_strategy == "linear_increase":
                timestep_weight = np.linspace(args.timestep_min_weight, args.timestep_max_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "linear_decrease":
                timestep_weight = np.linspace(args.timestep_max_weight, args.timestep_min_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "uniform":
                timestep_weight = np.ones(args.num_inference_steps)
            elif args.timestep_weight_strategy == "log_increase":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)

            elif args.timestep_weight_strategy == "log_decrease":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)
                timestep_weight = timestep_weight[::-1]

            print(f"timestep_weight: {timestep_weight}")

            prune_OBS_Diff(args, pipe, target_modules, device, prune_n=prune_n, prune_m=prune_m, timestep_weight=timestep_weight)
        
       
        elif args.prune_method == "OBS-Diff-Structured":
            if args.timestep_weight_strategy == "linear_increase":
                timestep_weight = np.linspace(args.timestep_min_weight, args.timestep_max_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "linear_decrease":
                timestep_weight = np.linspace(args.timestep_max_weight, args.timestep_min_weight, args.num_inference_steps)
            elif args.timestep_weight_strategy == "uniform":
                timestep_weight = np.ones(args.num_inference_steps)
            elif args.timestep_weight_strategy == "log_increase":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)

            elif args.timestep_weight_strategy == "log_decrease":
                linear_space = np.arange(0, args.num_inference_steps)
                timestep_weight = args.timestep_min_weight + (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)
                timestep_weight = timestep_weight[::-1]

            print(f"timestep_weight: {timestep_weight}")

            prune_OBS_Diff_Structured_SD15(args, pipe, device, timestep_weight=timestep_weight)
    if args.sparsity_type == "structured":
        prune_OBS_Diff_Structured_SD15(args, pipe, device, timestep_weight=timestep_weight)
        # if args.maxlayer == pipe.transformer.config.num_layers:
        #     args.maxlayer = pipe.transformer.config.num_layers - 1

    if args.sparsity_type != "structured":
        # sparsity_ratio = check_sparsity(pipe.transformer, target_modules)
        sparsity_ratio = check_sparsity(pipe.uniet, target_modules)

        print(f"sparsity sanity check {sparsity_ratio:.4f}")
    if args.sparsity_type == "structured":
        # check_size(pipe.transformer, target_modules)
        check_size(pipe.unet, target_modules)
    if args.demo_evaluate:
        height = 1024
        width = 1024
        num_inference_steps = 25
        guidance_scale = 7.0
        
        image = pipe(
            prompt="A cat holding a sign that says hello world",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(0)
        ).images[0] 
        os.makedirs("./eval_output", exist_ok=True)
        image.save(f"./eval_output/{args.demo_dir}")
        print(f"save image to ./eval_output/{args.demo_dir}")

    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        args.save_model = args.save_model + "/pruned_model.pth"
        torch.save(pipe.unet, args.save_model) 
        print(f"save model to {args.save_model}")

if __name__ == '__main__':
    main()