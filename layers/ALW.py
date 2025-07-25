import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from layers.Embed import PE

class ALW(nn.Module):
    """
    Adaptive Lookback Window Framework Driven by Wavelet Transform for Time Series Forecasting.
    """
    def __init__(self, configs):
        super(ALW, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.use_norm = configs.use_norm
        self.use_pe = configs.use_pe
        self.red_len = configs.red_len # Total length of the target prediction after wavelet decomposition.

        # Wavelet Transform Parameters Initialization
        # - J: Number of wavelet decomposition levels, controlling the multi-scale granularity
        #      of time series decomposition. Default is 3.
        # - wave: Wavelet type. 'db6' is suitable for processing smooth signals while
        #         preserving edge features, performing well in time series analysis.
        # - mode: Boundary padding mode. 'symmetric' padding can reduce distortion
        #         caused by boundary effects.
        self.J = configs.wavelet_levels if hasattr(configs, 'wavelet_levels') else 3
        self.wave = configs.wavelet if hasattr(configs, 'wavelet') else 'db6'
        self.mode = configs.pad_mode if hasattr(configs, 'pad_mode') else 'symmetric'
        self.dwt = DWT1DForward(wave=self.wave, J=self.J, mode=self.mode)
        self.idwt = DWT1DInverse(wave=self.wave)

        # Dynamically calculate input/output lengths for each scale component.
        # - A dummy input is used to perform wavelet decomposition and obtain
        #   the length of components at each scale.
        # - components_shapes: Lengths of the decomposed input sequence at various scales.
        # - length_shapes: Lengths of the decomposed prediction sequence at various scales.
        with torch.no_grad():
            dummy_enc = torch.zeros(1, 1, self.seq_len)
            yl_enc, yh_enc = self.dwt(dummy_enc)
            decomps_enc = [yl_enc] + yh_enc
            self.components_shapes = [comp.shape[-1] for comp in decomps_enc]

            dummy_pred = torch.zeros(1, 1, self.red_len)
            yl_pred, yh_pred = self.dwt(dummy_pred)
            decomps_pred = [yl_pred] + yh_pred
            self.length_shapes = [comp.shape[-1] for comp in decomps_pred]

        # Build predictors for each scale.
        # - predictors: A list of linear layers, mapping components from the input
        #   scale to the corresponding prediction scale.
        # - Weights are initialized to 1/in_len to ensure initial predictions do not
        #   overly amplify or attenuate the signal.
        self.predictors = nn.ModuleList()
        for in_len, out_len in zip(self.components_shapes, self.length_shapes):
            lin = nn.Linear(in_len, out_len)
            nn.init.constant_(lin.weight, 1 / in_len)
            nn.init.zeros_(lin.bias)
            self.predictors.append(lin)

        # Linear mappings for the time dimension, used to generate Q and K.
        # - q_linears and k_linears: Define linear layers for each scale to capture
        #   temporal correlations between time steps.
        # - Maps lengths to themselves to learn feature representations specific to the time dimension.
        self.q_linears = nn.ModuleList([
            nn.Linear(length, length) for length in self.components_shapes
        ])
        self.k_linears = nn.ModuleList([
            nn.Linear(length, length) for length in self.components_shapes
        ])

        # By learning weights along the channel dimension, the model can distinguish
        # which variables (channels) are more reliable or important for prediction
        # at specific frequencies/time scales, thus assigning higher weights during integration.
        self.importance_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, self.channels, length))
            for length in self.length_shapes
        ])

        # Positional Embeddings
        # - position_embeddings: Adds positional information to the prediction of each scale.
        # - Enhances the model's perception of temporal order, a common practice in time series models.
        self.position_embeddings = nn.ModuleList([
            PE(d_model=length)
            for length in self.length_shapes
        ])

        # Parameters alpha and beta
        # - alpha: Controls the sharpness of the soft-argmax. Initial value is 10.
        # - beta: Controls the steepness of the soft mask. Initial value is 10.
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 10)
            for _ in self.length_shapes
        ])
        self.beta = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 10)
            for _ in self.length_shapes
        ])

    def wavelet_decompose(self, x):
        """
        Wavelet decomposition function.
        Decomposes the input into low-frequency (yl) and high-frequency (yh) components,
        returning a multi-scale representation.
        """
        yl, yh = self.dwt(x)
        return [yl] + yh

    def compute_adaptive_windows(self, x):
        """
        Computes the soft truncation point for the adaptive lookback window at each scale.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, L] (batch, channels, length).

        Returns:
            decomposed (list): List of wavelet decomposed components.
            adaptive_windows (list): List of soft truncation points mu (μ) computed for each scale.
                                     Each element in the list has a shape of [B, C], representing
                                     the soft index of the optimal lookback window length
                                     for each sample and channel at that scale.
        """
        decomposed = self.wavelet_decompose(x)
        adaptive_windows = []

        for i, comp in enumerate(decomposed):
            B, C, L = comp.shape

            # Q/K Projection with Tanh Activation
            # Tanh rationale: Squashes the output values of Q/K projections into the range [-1, 1].
            # This helps normalize the inputs for subsequent attention score calculation,
            # potentially stabilizing training, and allows negative values to express
            # "unimportance" or "negative correlation."
            Q_proj = nn.Tanh()(self.q_linears[i](comp))  # [B,C,L]
            K_proj = nn.Tanh()(self.k_linears[i](comp))  # [B,C,L]

            # Compute Attention Scores
            # - Calculates the importance of each time step through the product of Q and K,
            #   scaled by sqrt(L) to prevent large inner products and unstable gradients.
            scores = torch.einsum("bcl,bcs->bls", Q_proj, K_proj) / (L ** 0.5)
            # - Tanh maps scores to [-1, 1], quantifying the information contribution.
            # - Mean aggregation along the time dimension, expanded back to original shape.
            weighted = torch.tanh(scores).mean(dim=1, keepdim=True).expand(-1, C, -1)  # [B, C, L]

            # Cumulative Reverse Sum
            # - Accumulates contributions backward from the most recent time step,
            #   simulating a review of history from the current point.
            # - A larger cumulative sum indicates greater importance of historical information
            #   up to that time step.
            # - The point with the largest cumulative contribution is identified as the
            #   best window truncation point.
            cum = torch.flip(torch.cumsum(torch.flip(weighted, [2]), dim=2), [2])  # [B,C,L]

            # Soft-argmax Calculation
            # 1. Uses softmax to convert cumulative weights into a probability distribution,
            #    where alpha controls the sharpness of the distribution.
            # 2. Computes a weighted average with position indices to obtain a continuous
            #    "soft" index position.
            # This soft index is differentiable, allowing optimization through gradient descent,
            # avoiding the non-differentiability of hard truncation.
            P = F.softmax(cum * self.alpha[i], dim=2)  # [B,C,L]
            idxs = torch.arange(L, device=comp.device).view(1,1,-1).expand(B,C,L)
            soft_idx = (P * idxs).sum(dim=2)  # [B,C]
            adaptive_windows.append(soft_idx)

        return decomposed, adaptive_windows

    def forward(self, x):
        # Optional Input Normalization
        # - Normalizes the input to reduce the impact of scale differences on
        #   decomposition and prediction.
        if self.use_norm:
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - mean) / std

        x_perm = x.permute(0, 2, 1)  # Adjust dimensions to [B, C, L]

        # Compute Wavelet Decomposed Components and Adaptive Window Soft Truncation Points (μ) for each scale.
        decomps, adaptive_windows = self.compute_adaptive_windows(x_perm)

        preds = []
        for i, (comp, win) in enumerate(zip(decomps, adaptive_windows)):
            B, C, L = comp.shape

            # Generate Time Step Indices
            positions = torch.arange(L, device=comp.device).view(1, 1, -1)

            # Create Soft Mask based on Adaptive Window Position
            # sigmoid(β * (t - μ)):
            # - When t < μ, it approaches 0 (distant past time steps are suppressed).
            # - When t > μ, it approaches 1 (recent time steps are retained).
            # - β controls the steepness of the transition; larger values make the mask
            #   closer to a hard truncation.
            mask = torch.sigmoid((positions - win.unsqueeze(-1)) * self.beta[i])  # [B,C,L]

            # Apply Mask to Component
            win_comp = comp * mask  # [B,C,L]

            # Predict and Add Positional Encoding
            # - The predictor generates the output, and PE provides temporal order information,
            #   enhancing the model's ability to perceive time structures.
            if self.use_pe:
                comp_pred = self.predictors[i](win_comp) + self.position_embeddings[i](C)
                # Learns which channels' prediction results are more reliable at this scale,
                # used for final weighted integration.
                comp_pred = F.softmax(self.importance_weights[i], dim=1) * comp_pred
            else:
                comp_pred = self.predictors[i](win_comp)
            preds.append(comp_pred)

        # Reconstruct the predicted results from various scales using Inverse Wavelet Transform.
        combined = self.idwt((preds[0], preds[1:]))

        # Return the results, including normalization parameters (if used).
        if self.use_norm:
            return combined, mean, std
        else:
            return combined, None, None