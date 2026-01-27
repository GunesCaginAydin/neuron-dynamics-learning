import torch
import torch.nn as nn
import torch.functional as F

def get_device(model):
    return model.wte.weight.device

def estimate_flops(self):
    """
    Return the estimated FLOPs per token for the model (forward + backward).
    Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
    Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
    On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
    With sliding windows, effective_seq_len varies per layer (capped by window size).
    Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
    This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
    - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
    - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
    """
    nparams = sum(p.numel() for p in self.parameters())
    # Exclude non-matmul params: embeddings and per-layer scalars
    value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
    nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                        self.resid_lambdas.numel() + self.x0_lambdas.numel())
    h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
    # Sum attention FLOPs per layer, accounting for sliding window
    attn_flops = 0
    for window_size in self.window_sizes:
        window = window_size[0]  # (left, right) tuple, we use left
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq
    num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
    return num_flops_per_token

def num_scaling_params(self):
    """
    Return all of the parameters, same as Chinchilla paper.
    Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
    But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
    My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
    Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
    Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
    """
    nparams = sum(p.numel() for p in self.parameters())
    return nparams

def compute_window_sizes(self, config):
    """
    Compute per-layer window sizes for sliding window attention.

    Returns list of (left, right) tuples for FA3's window_size parameter:
    - left: how many tokens before current position to attend to (-1 = unlimited)
    - right: how many tokens after current position to attend to (0 for causal)

    Pattern string is tiled across layers. Final layer always gets L (full context).
    Characters: L=long (full context), S=short (half context)
    """
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
    # Map characters to window sizes
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }
    # Tile pattern across layers
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    # Final layer always gets full context
    window_sizes[-1] = (long_window, 0)
    return window_sizes

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1