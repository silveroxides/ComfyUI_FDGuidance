import torch
import torch.nn.functional as F
import numpy as np

class FrequencyDecoupledGuidancePatcher:
    """
    An implementation of Frequency-Decoupled Guidance (FDG) from the paper
    "Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales"
    (arXiv:2506.19713). This node patches the model's CFG function to allow for
    separate guidance strengths on different image frequencies.

    This version uses a pure PyTorch implementation for Laplacian pyramid
    construction to avoid external dependencies and handle non-ideal image
    resolutions gracefully.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "levels": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "display": "slider"}),
                "freq_guidance_high": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 20.0, "step": 0.1, "display": "slider", "label": "High-Freq Guidance (w_high)"}),
                "freq_guidance_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "display": "slider", "label": "Low-Freq Guidance (w_low)"}),
                "apply_apg_projection": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled", "tooltip": "Apply Adaptive Projected Guidance (APG) to reduce oversaturation."}),
                "parallel_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "display": "slider", "tooltip": "Weight for the APG parallel component. Only used if APG is enabled."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Guidance/Patchers"


    def _get_gaussian_kernel(self, kernel_size: int, sigma: float, device, dtype):
        """Creates a 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= (kernel_size - 1) / 2.0
        g = coords**2
        g = (-(g / (2 * sigma**2))).exp()
        g /= g.sum()
        kernel = torch.outer(g, g)
        return kernel

    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0):
        """Applies a Gaussian blur to a tensor."""
        b, c, h, w = x.shape
        kernel = self._get_gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
        
        padding = (kernel_size - 1) // 2
        return F.conv2d(x, kernel, padding=padding, stride=1, groups=c)
        
    def _pyr_down(self, x: torch.Tensor):
        """Gaussian blur followed by downsampling."""
        blurred = self._gaussian_blur(x)
        return F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)

    def _pyr_up(self, x: torch.Tensor):
        """Upsampling followed by Gaussian blur."""
        # In classic pyramid implementations, the blur kernel is scaled by 4 during upsampling
        # to maintain brightness. We achieve a similar effect by blurring after upsampling.
        upsampled = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        return self._gaussian_blur(upsampled)

    def _build_laplacian_pyramid(self, tensor: torch.Tensor, levels: int):
        """Builds a Laplacian pyramid using pure PyTorch."""
        pyramid = []
        current_tensor = tensor
        for _ in range(levels - 1):
            down = self._pyr_down(current_tensor)
            up = self._pyr_up(down)
            
            # Ensure dimensions match before subtraction
            # This is the key fix for the original runtime error
            _, _, h, w = current_tensor.shape
            up_resized = F.interpolate(up, size=(h, w), mode='bilinear', align_corners=False)
            
            laplacian = current_tensor - up_resized
            pyramid.append(laplacian)
            current_tensor = down
            
        pyramid.append(current_tensor) # Add the smallest level (the "residual")
        return pyramid

    def _build_image_from_pyramid(self, pyramid: list):
        """Reconstructs an image from its Laplacian pyramid."""
        img = pyramid[-1] # Start with the smallest level
        for i in range(len(pyramid) - 2, -1, -1):
            up = self._pyr_up(img)
            
            # Ensure dimensions match before addition
            _, _, h, w = pyramid[i].shape
            up_resized = F.interpolate(up, size=(h, w), mode='bilinear', align_corners=False)
            
            img = up_resized + pyramid[i]
        return img
        

    def _interpolate(self, start, end, num_steps):
        if num_steps <= 1:
            return [start] if num_steps == 1 else []
        return np.linspace(start, end, num_steps).tolist()

    def _project(self, v0: torch.Tensor, v1: torch.Tensor):
        dtype = v0.dtype
        v0, v1 = v0.float(), v1.float()
        v1_norm = torch.nn.functional.normalize(v1.flatten(1), dim=-1).view_as(v1)
        v0_parallel = (v0 * v1_norm).sum(dim=[1, 2, 3], keepdim=True) * v1_norm
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def patch(self, model, enabled, levels, freq_guidance_high, freq_guidance_low, apply_apg_projection, parallel_weight):
        if not enabled or levels < 1:
            return (model,)

        guidance_scales = self._interpolate(freq_guidance_high, freq_guidance_low, levels)

        model_clone = model.clone()

        def frequency_domain_guidance(args):
            pred_cond_x0 = args["cond_denoised"]
            pred_uncond_x0 = args["uncond_denoised"]

            # Use the self-implemented pyramid builder
            pred_cond_pyramid = self._build_laplacian_pyramid(pred_cond_x0, levels)
            pred_uncond_pyramid = self._build_laplacian_pyramid(pred_uncond_x0, levels)

            pred_guided_pyramid = []
            for i, (p_cond, p_uncond) in enumerate(zip(pred_cond_pyramid, pred_uncond_pyramid)):
                diff = p_cond - p_uncond
                if apply_apg_projection:
                    diff_parallel, diff_orthogonal = self._project(diff, p_cond)
                    diff = parallel_weight * diff_parallel + diff_orthogonal

                scale = guidance_scales[i]
                p_guided = p_uncond + scale * diff
                pred_guided_pyramid.append(p_guided)

            # Use the self-implemented pyramid reconstruction function
            guided_x0 = self._build_image_from_pyramid(pred_guided_pyramid)

            # Recombine the guided result into the final prediction
            x0_diff = pred_cond_x0 - pred_uncond_x0
            pred_diff = args["cond"] - args["uncond"]
            
            # Avoid division by zero
            scaling_factor = pred_diff / (x0_diff + 1e-9)
            
            new_x0_diff = guided_x0 - pred_uncond_x0
            new_pred_diff = new_x0_diff * scaling_factor
            
            final_pred = args["uncond"] + new_pred_diff
            return final_pred.to(args["cond"].dtype)

        model_clone.set_model_sampler_cfg_function(frequency_domain_guidance)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "FrequencyDecoupledGuidancePatcher": FrequencyDecoupledGuidancePatcher
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrequencyDecoupledGuidancePatcher": "FDG (Frequency Decoupled Guidance)"
}
