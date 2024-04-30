import torch

from ..modules.attention import *
from ..modules.diffusionmodules.util import (AlphaBlender, linear,
                                             timestep_embedding)


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)
  
        ## False
        if disable_temporal_crossattention:
            print(f"disable_temporal_crossattention: {disable_temporal_crossattention}")
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa
        # print(f"switch_temporal_ca_to_sa: {switch_temporal_ca_to_sa}")  
        ## False

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None, 
        time_cross_attn=False, h=None, w=None
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps, time_cross_attn, h, w)
        else:
            return self._forward(x, context, timesteps, time_cross_attn, h=h, w=w)

    def _forward(self, x, context=None, timesteps=None, 
                 time_cross_attn=False, h=None, w=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape    ## B=2*num_frames
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        attn = None
        # attn_store = {"cross":[], "self": []}
        attn_store = {}
        attn_type = 'time_context cross' if context is not None and time_cross_attn else 'spatial_context cross'
        
        use_attn = True if context is not None and time_cross_attn else False
        
        if self.disable_self_attn:
            print(f"\nVideoTransformerBlock.attn1 is a {attn_type}-attention")
            x_attn = self.attn1(self.norm1(x), context=context, 
                                      use_attn=use_attn)
            if use_attn:
                x_attn, attn = x_attn
                
                key = "cross"  ##if use_attn else "cross_with_spatial"
                if key in attn_store:
                    attn_store[key].append(attn)
                else:
                    attn_store[key] = [attn]
            
            x = x_attn + x
            # del attn  ## TODO: check validity
        else:
            print(f"\nVideoTransformerBlock.attn1 is a temporal self-attention")
            x = self.attn1(self.norm1(x)) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                print("\nVideoTransformerBlock.attn2 is a self-attention")
                x = self.attn2(self.norm2(x)) + x
            else:
                print(f"\nVideoTransformerBlock.attn2 is {attn_type}-attn")
                ## [N, 77, 1024]
                x_attn = self.attn2(self.norm2(x), context=context, use_attn=use_attn)
                
                if use_attn:
                    x_attn, attn = x_attn

                    ## Rearrange attn
                    attn = rearrange(  ## (2 h w), 25, 77
                        attn, "(b s) t c -> (b t) s c", 
                        s=S,  ## h * w
                        b=B // timesteps,  ## B = 2 * num_frames
                        c=attn.shape[-1],  ## 77
                        t=timesteps,  ## num_frames
                    )  ## (2 25), (h w), 77
                    
                    if h is not None and w is not None:
                        attn = rearrange(
                            attn, "b (h w) c -> b c h w",
                            h=h, w=w, b=attn.shape[0], c=attn.shape[-1])
                        print(f"Spatial rearranged cross-attn map: {attn.shape}")
                    
                    key = "cross"  ## if use_attn else "cross_with_spatial"
                    if key in attn_store:
                        attn_store[key].append(attn)
                    else:
                        attn_store[key] = [attn]
                    # del attn
                
                x = x_attn + x
                
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", 
            s=S,  ## h * w
            b=B // timesteps,  ## B = 2 * num_frames
            c=C, 
            t=timesteps,  ## num_frames
        )

        if len(attn_store) > 0 :
            return x, attn_store
        
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,  ## FIXME: currently control both temporal and spatial (super module)
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    dim=inner_dim,
                    n_heads=n_time_mix_heads,
                    d_head=time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,  ## 0.5
            merge_strategy=merge_strategy  ## "learned_with_images"
        )
        # self.attn_store = {}
    
    # def reset(self):
    #     # self.step_store = self.get_empty_store()
    #     self.attn_store = {}

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        **kwargs,  ##  place in UNet
    ) -> torch.Tensor:

        self.attn_store = {}

        _, _, h, w = x.shape
        x_in = x  ## Residual ate
        spatial_context = None
        if exists(context):
            spatial_context = context
            print(f"spatial_context's shape: {spatial_context.shape}")

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            print("Take spatial context as time_context")
            time_cross_attn = False

            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            print("Input time_context's shape: ", time_context.shape)  ## [2, 77, 1024]
            time_cross_attn = True

            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            print(f"Repeated by: h={h}, w={w}, n=h*w={h*w}, time_context's shape={time_context.shape}")
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")
            print("Final time_context's shape: ", time_context.shape, )  ## [1152, 77, 1024]

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(  ## Create sinusoidal timestep embeddings.
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_pos_embed(t_emb)  ## MLP(SiLU(MLP(t_emb)))
        emb = emb[:, None, :]
        
        if "key" in kwargs:
            place_in_unet = kwargs["key"]  ## 'down', 'mid', 'up'
            key = f"{place_in_unet}_{'temporal'}"
        
        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            ## 1. Spatial Transformer: self and cross
            x = block(
                x, context=spatial_context,
            )
            if isinstance(x, tuple):
                x, spatial_attn_store = x
                ## TODO: save spatial_attn_store
            
            ## 2. Temporal Transformer: self and cross
            x_mix = x
            x_mix = x_mix + emb
            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps, 
                              time_cross_attn=time_cross_attn, h=h, w=w
            )
            if isinstance(x_mix, tuple):
                x_mix, time_attn_store = x_mix
                for k in time_attn_store:  ## 'cross', 'self'
                    self.attn_store[f"{key}_{k}"] = time_attn_store[k]
                
                # del time_attn_store
            
            ## 3. Alpha Blend Spatial and Temporal
            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
                image_only_indicator=image_only_indicator,
            )  ##  = alpha * spatial_x + (1 - alpha) * temporal_x; alpha ≈ sigmoid(factor)
        
        ## Output projection and reshape: ====
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out
