import math
import torch
import torch.nn as nn

from .OBS_Diff import OBS_Diff


class OBS_Diff_UNet(OBS_Diff):
    """
    Extends OBS_Diff so it can also be used on nn.Conv2d layers, which is
    what SDXL's UNet is mostly built from (the base OBS_Diff class only
    reshapes/transposes correctly for nn.Linear / transformers.Conv1D,
    since it was written for the MMDiT / pure-transformer setting).

    For a conv layer, treat each output pixel as a linear function of the
    im2col patch that produced it: [C_in * kH * kW] -> [C_out]. Unfolding
    the input the same way conv2d does internally turns the conv layer
    into exactly the same "X X^T" reconstruction problem OBS/SparseGPT
    solve for Linear layers, so the rest of the pipeline (fasterprune)
    needs no changes at all.
    """

    def add_batch(self, inp, out, W_new):
        if not isinstance(self.layer, nn.Conv2d):
            # Linear / Conv1D: identical to the base class.
            super().add_batch(inp, out, W_new)
            return

        if len(inp.shape) == 3:
            inp = inp.unsqueeze(0)

        unfold = nn.Unfold(
            kernel_size=self.layer.kernel_size,
            dilation=self.layer.dilation,
            padding=self.layer.padding,
            stride=self.layer.stride,
        )
        # inp: [B, C_in, H, W] -> patches: [B, C_in*kH*kW, L]
        patches = unfold(inp)
        # -> [C_in*kH*kW, B*L]   (each column = one im2col patch = one "sample")
        patches = patches.permute(1, 0, 2).reshape(patches.shape[1], -1)

        W_old = self.sum_weight
        W_total = W_old + W_new
        self.H *= W_old / W_total
        self.sum_weight = W_total

        norm_factor = math.sqrt(2 / self.sum_weight)
        patches = norm_factor * patches.float()
        self.H += patches.matmul(patches.t())
