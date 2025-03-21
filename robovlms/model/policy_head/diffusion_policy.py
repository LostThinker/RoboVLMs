import math
import torch
from torch import nn
from typing import Optional, Tuple, Union
from einops import rearrange, reduce
from torch.nn import functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import functools
from robovlms.model.policy_head.base_policy import (
    BasePolicyHead,
    FCDecoder,
    LSTMDecoder,
    GPTDecoder,
)


class ModuleAttrMixin(nn.Module):
    """
    Borrowed from https://github.com/real-stanford/diffusion_policy
    """

    def __init__(self):
        super().__init__()
        # self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb


def SinusoidalPosEmb(x, dim):
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


def _get_default_config(name):
    _default_configs = {
        "noise_scheduler_configs": dict(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",  # Yilun's paper uses fixed_small_log instead, but easy to cause Nan
            clip_sample=True,  # required when predict_epsilon=False
            prediction_type="epsilon",  # or sample
        ),
        "enc_configs": dict(
            n_layer=0,  # 0 means for MLP encoders
            n_head=8,
            p_drop_attn=0.1,
        ),
        "dec_configs": dict(
            n_layer=12,
            n_head=8,
            p_drop_attn=0.1,
        ),
    }
    return _default_configs.get(name, None)


class TransformerForDiffusion(ModuleAttrMixin):
    """
    Modified from diffusion policy: https://github.com/real-stanford/diffusion_policy

    The original version takes encoder-decoder structure to fuse conditions and input sequence. While here,
    our conditional diffusion models support two modes: bert-like and encoder-decoder.

    For bert-like structure:
        conditions will be concatenated to the input sequence, which, e.g., is typically Gaussian inputs
        at timestep 0 in diffusion models. Information flows from the conditions to the input sequence
        through bi-directional attentions.
    For encoder-decoder structure:
        conditions will be processed by the encoder first, and then fused by cross attention with the input
        sequence in the decoder.

    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        action_dim: int,
        fwd_pred_next_n: int,
        latent: int = 1,
        enc_configs: Optional[Union[dict, str]] = None,
        dec_configs: Optional[Union[dict, str]] = None,
        p_drop_emb: float = 0.1,
        # the following three should be carefully set
        causal_attn: bool = True,
        separate_gripper: bool = False,
    ) -> None:

        super().__init__()

        if enc_configs == "default":
            enc_configs = _get_default_config("enc_configs")
        if dec_configs is None or dec_configs == "default":
            dec_configs = _get_default_config("dec_configs")

        self.latent = latent
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.fwd_pred_next_n = fwd_pred_next_n
        T = fwd_pred_next_n

        # condition on diffusion timestep and observations
        n_cond_tok = 1 + latent

        # input embedding stem
        self.input_emb = nn.Linear(self.action_dim, self.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, self.hidden_size))
        self.drop = nn.Dropout(p_drop_emb)
        # self.drop = functools.partial(nn.functional.dropout1d, p=p_drop_emb)

        # cond encoder
        # self.time_emb = SinusoidalPosEmb(self.hidden_size)
        self.time_emb = functools.partial(SinusoidalPosEmb, dim=self.hidden_size)
        self.cond_obs_emb = nn.Linear(self.in_features, self.hidden_size)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None

        if enc_configs is not None:
            self.cond_pos_emb = nn.Parameter(
                torch.zeros(1, n_cond_tok, self.hidden_size)
            )
            n_cond_layers = enc_configs["n_layer"]
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=enc_configs["n_head"],
                    dim_feedforward=4 * self.hidden_size,
                    dropout=enc_configs["p_drop_attn"],
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(self.hidden_size, 4 * self.hidden_size),
                    nn.Mish(),
                    nn.Linear(4 * self.hidden_size, self.hidden_size),
                )

            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=dec_configs["n_head"],
                dim_feedforward=4 * self.hidden_size,
                dropout=dec_configs["p_drop_attn"],
                activation="gelu",
                batch_first=True,
                norm_first=True,  # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=dec_configs["n_layer"]
            )

        else:
            self.encoder = None

            # the decoder is initialized using the bert-like structure.
            # all tokens of obs conditions, time embeddings, and diffusion inputs are concatenated,
            # which are then processed by bi-directional attention for information fusing.
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=dec_configs["n_head"],
                dim_feedforward=4 * self.hidden_size,
                dropout=dec_configs["p_drop_attn"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(
                decoder_layer=decoder_layer, num_layers=dec_configs["n_layer"]
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            S = n_cond_tok
            t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
            mask = t >= (
                s - 1
            )  # add one dimension since time is the first token in cond
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("memory_mask", mask)

        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.separate_gripper = separate_gripper
        if self.separate_gripper:
            self.head = FCDecoder(
                in_features, hidden_size, action_dim, down_sample="none", latent=latent
            )
        else:
            self.head = nn.Linear(self.hidden_size, self.action_dim)
        # self.action_head = nn.Linear(self.hidden_size, self.action_dim - 1)
        # self.gripper_head = nn.Linear(self.hidden_size, 1)

        # constants
        self.T = T
        self.n_cond_tok = n_cond_tok
        self.horizon = self.fwd_pred_next_n

        # init
        self.apply(self._init_weights)
        print("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        sample_masks: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        sample_masks: (B,T)
        cond_masks: (B,T')
        """
        # print(sample.shape, timestep.shape, cond.shape)
        bs = sample.shape[0]
        _, cond_len = cond.shape[:2]
        assert cond_len == self.latent

        if sample_masks is None:
            sample_masks = torch.ones(sample.shape[:-1], dtype=torch.bool).to(
                sample.device
            )
        if cond_masks is None:
            cond_masks = torch.ones(cond.shape[:-1], dtype=torch.bool).to(sample.device)

        # map mask values: 1 -> 0, 0 -> 1, to fit the inferface of torch transformer encoder / decoder
        sample_masks = ~sample_masks
        cond_masks = ~cond_masks

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        # (B,1,n_emb)
        time_emb = self.time_emb(timesteps).unsqueeze(1).to(sample.dtype)

        cond_embeddings = time_emb
        cond_time_masks = torch.zeros(size=(bs, 1)).bool().to(sample.device)
        if cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)
            # (B,To,n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            cond_time_masks = torch.cat([cond_time_masks, cond_masks], dim=1)
        tc = cond_embeddings.shape[1]

        # process input
        input_emb = self.input_emb(sample)
        if self.encoder is None:
            # BERT
            token_embeddings = torch.cat([cond_embeddings, input_emb], dim=1)
            token_masks = torch.cat([cond_time_masks, sample_masks], dim=1)

            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,Tc+T,n_emb)
            x = self.decoder(src=x, mask=self.mask, src_key_padding_mask=token_masks)
            # (B,T,n_emb)
            x = x[:, tc:, :]
        else:
            # encoder
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            if isinstance(self.encoder, nn.TransformerEncoder):
                x = self.encoder(x, src_key_padding_mask=cond_time_masks)
            else:
                x = self.encoder(x)
            memory = x
            # (B,n_cond_tok,n_emb)

            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
                tgt_key_padding_mask=sample_masks,
                memory_key_padding_mask=cond_time_masks,
            )
            # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        if self.separate_gripper:
            x = torch.cat(x, dim=-1)
        return x


class DiffusionPolicy(BasePolicyHead):
    def __init__(
        self,
        in_features: int,
        action_dim: int,
        down_sample: str = "pooling",
        latent: int = 1,
        fwd_pred_next_n: int = 1,
        window_size: int = 1,
        diffusion_trf_configs: Optional[dict] = None,
        noise_scheduler_configs: Optional[dict] = None,
        num_inference_steps: int = 100,
        separate_gripper: bool = False,
        history_encoder: str = None,
        **kwargs,
    ):

        # default hidden_size is in_features
        if "hidden_size" not in kwargs:
            kwargs.update({"hidden_size": in_features})
        kwargs.update({"hidden_size": in_features})
        super().__init__(
            action_dim=action_dim, down_sample=down_sample, latent=latent, **kwargs
        )
        self.in_features = in_features
        self.fwd_pred_next_n = fwd_pred_next_n
        self.global_pooling_1d = nn.AdaptiveAvgPool1d(latent)
        self.num_inference_steps = num_inference_steps
        self.separate_gripper = separate_gripper
        self.window_size = window_size

        if noise_scheduler_configs is None:
            noise_scheduler_configs = _get_default_config("noise_scheduler_configs")
        self.noise_scheduler = DDPMScheduler(**noise_scheduler_configs)

        if diffusion_trf_configs is None:
            diffusion_trf_configs = {
                "in_features": in_features,
                "hidden_size": self.hidden_size,
                "action_dim": action_dim,
                "fwd_pred_next_n": fwd_pred_next_n,
                "latent": latent,
                "enc_configs": "default",
                "dec_configs": "default",
                "separate_gripper": self.separate_gripper,
            }
        self.model = TransformerForDiffusion(**diffusion_trf_configs)

        hidden_size = kwargs["hidden_size"]
        if history_encoder is None:
            self.hist_enc = nn.Identity()
        elif history_encoder == "lstm":
            self.hist_enc = LSTMDecoder(
                in_features,
                action_dim,
                down_sample,
                latent,
                fwd_pred_next_n,
                window_size,
                policy_rnn_dropout_p=0.0,
                early_return=True,
                hidden_size=hidden_size,
            )
        elif history_encoder == "gpt":
            self.hist_enc = GPTDecoder(
                in_features,
                action_dim,
                down_sample,
                latent,
                fwd_pred_next_n,
                window_size,
                early_return=True,
                hidden_size=hidden_size,
            )
        else:
            raise NotImplementedError

    def _downsample(self, x):
        """
        x: b x n_patch x d
        """
        if self.down_sample == "pooling":
            return self.global_pooling_1d(x.transpose(1, 2)).transpose(1, 2)
        elif self.down_sample == "none":
            return x
        else:
            raise NotImplementedError(
                f"Downsample method {self.down_sample} not implemented"
            )

    def get_labels(self, pred_actions, labels, action_masks, tok_seq, **kwargs):
        """
        pred_actions: not used
        labels: b x l x chunk_size x act_dim
        tok_seq: b x l x n_patch x d
        """
        # import pdb; pdb.set_trace()
        assert isinstance(labels, (tuple, list))
        if labels[1].ndim != labels[0].ndim:
            labels = (labels[0], labels[1].unsqueeze(-1))
        labels = torch.cat([labels[0], labels[1]], dim=-1)

        if action_masks is None:
            action_masks = torch.ones(labels.shape[:-1], dtype=torch.bool).to(
                labels.device
            )
        else:
            action_masks = action_masks.bool()

        assert labels.dim() in {3, 4} and tok_seq.dim() in {3, 4}

        input_dim = labels.dim()
        tok_seq = self._downsample(tok_seq)
        tok_seq = self.hist_enc(tok_seq)
        if input_dim > tok_seq.ndim:
            tok_seq = tok_seq.unsqueeze(2)

        if input_dim == 4:
            bs, seq_len, chunk_size, act_dim = labels.shape
            _, _, n_patch, hidden_size = tok_seq.shape
            labels = rearrange(labels, "b l n d -> (b l) n d")
            tok_seq = rearrange(tok_seq, "b l n d -> (b l) n d")
            action_masks = rearrange(action_masks, "b l n -> (b l) n")
        else:
            bs, chunk_size, act_dim = labels.shape
            _, n_patch, hidden_size = tok_seq.shape
            seq_len = 1
        # print(self.hidden_size, hidden_size)
        assert self.hidden_size == hidden_size, self.action_dim == act_dim

        # follow the conventional name as the original code
        cond = tok_seq  # b*l x n_patch x d
        trajectory = labels  # b*l x chunk_size x act_dim

        # downsample the input conditions
        # cond = self._downsample(cond)
        # cond = self.hist_enc(cond)
        assert cond.shape[1] == self.latent

        # Sample noise that we'll add to the images
        noise = (
            torch.randn(trajectory.shape, device=trajectory.device)
            .to(tok_seq.device)
            .to(tok_seq.dtype)
        )
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # b*l x chunk_size x act_dim
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        ).to(tok_seq.dtype)
        noisy_trajectory[~action_masks] = 0

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond, action_masks)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if input_dim == 4:
            pred = rearrange(pred, "(b l) n d -> b l n d", b=bs, l=seq_len)
            target = rearrange(target, "(b l) n d -> b l n d", b=bs, l=seq_len)
            action_masks = rearrange(action_masks, "(b l) n -> b l n", b=bs, l=seq_len)

        return pred, target, action_masks

    def _forward_train(self, tok_seq, **kwargs):
        # during training, we do not need to feedforward anything. the predictions and labels are
        # generated and processed in self.get_labels
        return None, None

    def conditional_sample(
        self, sample_shape, cond, sample_masks=None, generator=None, **kwargs
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=sample_shape, dtype=cond.dtype, device=cond.device, generator=generator
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            model_output = model(trajectory, t, cond, sample_masks=sample_masks)

            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        return trajectory

    def _forward_test(self, tok_seq, **kwargs):
        """
        tok_seq: b x l x n x d or b x n x d
        """
        # import pdb; pdb.set_trace()
        assert tok_seq.dim() in {3, 4}
        input_dim = tok_seq.dim()
        tok_seq = self._downsample(tok_seq)
        tok_seq = self.hist_enc(tok_seq)
        if input_dim > tok_seq.ndim:
            tok_seq = tok_seq.unsqueeze(2)

        if tok_seq.dim() == 4:
            bs, seq_len, n_patch, hidden_size = tok_seq.shape
            tok_seq = rearrange(tok_seq, "b l n d -> (b l) n d")
        else:
            bs, n_patch, hidden_size = tok_seq.shape
            seq_len = 1

        cond = tok_seq
        # cond = self._downsample(cond)
        # cond = self.hist_enc(cond)
        # import pdb; pdb.set_trace()
        assert cond.shape[1] == self.latent

        sample_shape = (bs * seq_len, self.fwd_pred_next_n, self.action_dim)

        # run sampling
        actions = self.conditional_sample(sample_shape, cond)

        grippers = actions[..., -1:]
        actions = actions[..., :-1]

        if input_dim == 4:
            actions = rearrange(actions, "(b l) n d -> b l n d", b=bs, l=seq_len)
            grippers = rearrange(grippers, "(b l) n d -> b l n d", b=bs, l=seq_len)

        return actions, grippers

    def forward(self, tok_seq, **kwargs):
        if self.training:
            return self._forward_train(tok_seq, **kwargs)
        else:
            return self._forward_test(tok_seq, **kwargs)

    def loss(self, pred_action, labels, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(labels.shape[:-1], dtype=torch.bool).to(
                labels.device
            )
        attention_mask = attention_mask.unsqueeze(-1)
        loss = F.mse_loss(pred_action, labels, reduction="none")
        loss = loss * attention_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return {"loss_arm": loss, "loss_gripper": 0, "acc_gripper": 0.0}


if __name__ == "__main__":
    in_features = 1024
    action_dim = 7
    down_sample = "none"
    fwd_next_n = 10
    bs, seq_len = 2, 3
    model = DiffusionPolicy(
        in_features,
        action_dim,
        down_sample,
        fwd_pred_next_n=fwd_next_n,
        window_size=seq_len,
        history_encoder="lstm",
    )

    device = "cuda:0"
    tok_seq = torch.randn(bs, seq_len, 1, in_features).to(device)
    action_arm = torch.randn(bs, seq_len, fwd_next_n, action_dim - 1).to(device)
    action_gripper = torch.ones(bs, seq_len, fwd_next_n, 1).to(device)
    model = model.to(device)
    model.training = False
    action_masks = torch.ones(bs, seq_len, fwd_next_n).long().to(device)
    pred_actions = None
    labels = (action_arm, action_gripper)
    # action, gripper = model(tok_seq)
    pred, target, action_masks = model.get_labels(
        pred_actions, labels, action_masks, tok_seq
    )
    import pdb

    pdb.set_trace()
