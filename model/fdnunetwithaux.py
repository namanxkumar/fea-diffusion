from functools import partial
from typing import List, Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def exists(value):
    return value is not None


def zero_out(module: nn.Module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def divisible_by(value, divisor):
    return value % divisor == 0


class Downsample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        downsample_factor = 2
        self.downsample = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (c p1 p2) h w",
                p1=downsample_factor,
                p2=downsample_factor,
            ),
            nn.Conv2d(input_dim * (downsample_factor**2), output_dim, 1),
        )

    def forward(self, x: Tensor):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        upsample_factor = 2
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsample_factor, mode="nearest"),
            nn.Conv2d(
                input_dim, output_dim, 3, padding=1
            ),  # [gen] Convolution layer after the upsample is to smooth out the upsampled image
        )

    def forward(self, x: Tensor):
        return self.upsample(x)


class FeatureDenormalization(nn.Module):
    def __init__(
        self, input_dim: int, feature_dim: int, num_groups_for_normalization: int = 8
    ):
        super().__init__()

        self.normalize = nn.GroupNorm(
            num_groups_for_normalization, input_dim, affine=False
        )
        self.scale_convolution = nn.Conv2d(
            feature_dim, input_dim, kernel_size=3, padding=1
        )
        self.shift_convolution = nn.Conv2d(
            feature_dim, input_dim, kernel_size=3, padding=1
        )
        self.activation = nn.SiLU()

    def forward(self, x: Tensor, features: Tensor):
        assert (
            features.size()[2:] == x.size()[2:]
        ), "features must have the same spatial dimensions as x"
        x = self.normalize(x)
        scale = self.scale_convolution(features)
        shift = self.shift_convolution(features)
        x = x * (scale + 1) + shift
        return self.activation(x)


class ConditionFeatureExtractor(nn.Module):
    def __init__(self, num_condition_channels: int, stagewise_dimensions: List[int]):
        super().__init__()

        stagewise_input_to_output_dims = list(
            zip(stagewise_dimensions[:-1], stagewise_dimensions[1:])
        )

        self.pre_extractors = nn.Sequential(
            nn.Conv2d(num_condition_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        128,
                        stagewise_input_to_output_dims[0][0],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.SiLU(),
                ),
            ]
        )

        for index, (input_dim, output_dim) in enumerate(stagewise_input_to_output_dims):
            module = nn.Sequential(
                (
                    nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
                    if (index == len(stagewise_input_to_output_dims) - 1)
                    else Downsample(input_dim, output_dim)
                ),
                nn.SiLU(),
            )
            self.extractors.append(module)

    def forward(self, x: Tensor):
        x = self.pre_extractors(x)

        output_features = []

        for index in range(len(self.extractors)):
            x = self.extractors[index](x)
            output_features.append(x)

        return output_features


class LeanResnetSubBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.project = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.activate = nn.SiLU()

    def forward(self, x: Tensor, scale_shift=None):
        x = self.project(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activate(x)
        return x


class ResnetSubBlock(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_groups_for_normalization: int = 8
    ):
        super().__init__()
        self.project = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.normalize = nn.GroupNorm(num_groups_for_normalization, output_dim)
        self.activate = nn.SiLU()

    def forward(self, x: Tensor, scale_shift=None):
        x = self.project(x)
        x = self.normalize(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activate(x)
        return x


class ConditionedResnetBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_features_dim: Optional[int] = None,
        use_fdn: bool = True,
        time_embedding_dim: Optional[int] = None,
        num_groups_for_normalization: int = 8,
    ):
        super().__init__()
        self.use_fdn = use_fdn

        if self.use_fdn:
            assert exists(
                condition_features_dim
            ), "condition features dim must be provided"

        self.time_embedding_to_scale_shift = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, output_dim * 2))
            if exists(time_embedding_dim)
            else None
        )

        self.conditional_block1 = (
            FeatureDenormalization(
                input_dim,
                condition_features_dim,
                num_groups_for_normalization=num_groups_for_normalization,
            )
            if use_fdn
            else None
        )

        self.block1 = (
            LeanResnetSubBlock(input_dim, output_dim)
            if use_fdn
            else ResnetSubBlock(
                input_dim,
                output_dim,
                num_groups_for_normalization=num_groups_for_normalization,
            )
        )

        self.conditional_block2 = (
            FeatureDenormalization(
                output_dim,
                condition_features_dim,
                num_groups_for_normalization=num_groups_for_normalization,
            )
            if use_fdn
            else None
        )

        self.block2 = (
            LeanResnetSubBlock(output_dim, output_dim)
            if use_fdn
            else ResnetSubBlock(
                output_dim,
                output_dim,
                num_groups_for_normalization=num_groups_for_normalization,
            )
        )

        self.residual_convolution = (
            nn.Conv2d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_features: Optional[torch.Tensor] = None,
        time_embedding: Optional[Tensor] = None,
    ):
        if self.use_fdn:
            assert exists(condition_features), "condition features must be provided"

        h = x

        if exists(self.conditional_block1):
            h = self.conditional_block1(x, condition_features)

        scale_shift = None
        if exists(self.time_embedding_to_scale_shift) and exists(time_embedding):
            time_embedding = self.time_embedding_to_scale_shift(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1")
            scale_shift = time_embedding.chunk(
                2, dim=1
            )  # split into scale and shift each of dimension b c/2 1 1

        h = self.block1(h, scale_shift=scale_shift)

        if exists(self.conditional_block2):
            h = self.conditional_block2(h, condition_features)

        h = self.block2(h)

        return h + self.residual_convolution(x)


class AuxiliaryRangePredictor(nn.Module):
    """
    Auxiliary Range Predictor Module

    Takes input from the output of the middle block and predicts the range of the output image by flattening the spatial dimensions and passing through a few linear layers.

    Args:
        input_dim: int, input dimension
        output_dim: int, output dimension
        num_groups_for_normalization: int, number of groups for group normalization

    Returns:
        Tensor: output tensor
    """

    def __init__(
        self,
        input_dim: int,
        middle_height: int,
        middle_width: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        self.flatten = Rearrange(
            "b c h w -> b (c h w)", h=middle_height, w=middle_width, c=input_dim
        )

        self.layers = nn.Sequential(
            nn.Linear(
                input_dim * middle_height * middle_width,
                hidden_dim,
            ),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(num_layers - 2)
            ],
        )

        self.sigmoid_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        self.log_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor):
        x = self.flatten(x)
        x = self.layers(x)
        return self.sigmoid_output_layer(x), self.log_output_layer(x)


class FDNUNetEncoder(nn.Module):
    """
    FDNUNet with Auxiliary Range Prediction

    Note: Model outputs absolute value of ranges, take the negative of even indices to get the range
    """

    def __init__(
        self,
        input_dim: int,
        # image_height: int,
        # image_width: int,
        # range_prediction_hidden_dim: int = 256,
        # range_prediction_num_layers: int = 3,
        initial_dim: Optional[int] = None,
        # final_dim: Optional[int] = None,
        num_stages: int = 4,
        num_channels: int = 3,
        num_auxiliary_condition_channels: int = 3,
        num_condition_channels: Optional[int] = None,
        resnet_num_groups_for_normalization: int = 8,
        # disable_auxiliary=False,
        # only_auxiliary=False,
        # positional_embedding_theta: int = 10000,
        # attention_head_dim: int = 32,
        # num_attention_heads: int = 4,
        # use_full_attention: bool = False,  # defaults to full attention only for inner most layer
        # use_flash_attention: bool = False,
    ):
        super().__init__()

        # Define Dimensions
        self.num_stages = num_stages

        stagewise_dim_multipliers = tuple([2**i for i in range(num_stages)])

        self.num_channels = num_channels

        self.num_condition_channels = num_condition_channels

        if exists(num_condition_channels):
            input_channels = num_channels + num_condition_channels
        else:
            input_channels = num_channels

        initial_dim = initial_dim if exists(initial_dim) else input_dim

        stagewise_dimensions = [
            initial_dim,
            *map(lambda multiplier: input_dim * multiplier, stagewise_dim_multipliers),
        ]
        assert (
            len(stagewise_dimensions) == num_stages + 1
        ), "stagewise dimensions must be equal to number of stages + 1"

        stagewise_input_to_output_dims = list(
            zip(stagewise_dimensions[:-1], stagewise_dimensions[1:])
        )
        assert (
            len(stagewise_input_to_output_dims) == num_stages
        ), "stagewise input to output dimensions must be equal to number of stages"

        # # Define Time Embedding
        # time_embedding_dim = input_dim * 4

        # self.time_embedding_mlp = nn.Sequential(
        #     SinusoidalPosEmb(input_dim, theta=positional_embedding_theta),
        #     nn.Linear(input_dim, time_embedding_dim),
        #     nn.GELU(),
        #     nn.Linear(time_embedding_dim, time_embedding_dim),
        # )

        # Define Condition Feature Extractor
        self.condition_feature_extractor = ConditionFeatureExtractor(
            num_auxiliary_condition_channels, stagewise_dimensions
        )

        # Define Resnet
        # resnet_module = partial(ConditionedResnetBlock, num_groups_for_normalization = resnet_num_groups_for_normalization, time_embedding_dim = time_embedding_dim)
        resnet_module = partial(
            ConditionedResnetBlock,
            num_groups_for_normalization=resnet_num_groups_for_normalization,
        )
        # resnet_module_without_fdn = partial(
        #     ConditionedResnetBlock,
        #     use_fdn=False,
        #     num_groups_for_normalization=resnet_num_groups_for_normalization,
        # )

        # # Define Attention
        # if not use_full_attention:
        #     stagewise_use_full_attention = (*((False,) * (num_stages - 1)), True)
        # else:
        #     stagewise_use_full_attention = (True,) * num_stages

        # stagewise_num_attention_heads = (num_attention_heads,) * num_stages
        # stagewise_attention_head_dim = (attention_head_dim,) * num_stages

        # assert (
        #     len(stagewise_use_full_attention)
        #     == len(stagewise_num_attention_heads)
        #     == len(stagewise_attention_head_dim)
        #     == num_stages
        # ), "number of stages must be equal for attention parameters"

        # full_attention_module = partial(
        #     FullAttention, use_flash_attention=use_flash_attention
        # )

        # self.disable_auxiliary = disable_auxiliary

        # self.only_auxiliary = only_auxiliary

        # assert not (
        #     self.disable_auxiliary and self.only_auxiliary
        # ), "Cannot disable and only use auxiliary"

        ## Build Layers
        self.initial_convolution = nn.Conv2d(input_channels, initial_dim, 3, padding=1)

        self.down_layers = nn.ModuleList([])

        # if not self.only_auxiliary:
        #     self.up_layers = nn.ModuleList([])

        # Define Downsampling Layers
        for index, (input_dim, output_dim) in enumerate(stagewise_input_to_output_dims):
            is_last = index == (num_stages - 1)

            # attention_module = (
            #     full_attention_module if use_full_attention else LinearAttention
            # )
            downsample_module = (
                Downsample
                if not is_last
                else partial(nn.Conv2d, kernel_size=3, padding=1)
            )

            self.down_layers.append(
                nn.ModuleList(
                    [
                        resnet_module(
                            input_dim, input_dim, condition_features_dim=input_dim
                        ),
                        resnet_module(
                            input_dim, input_dim, condition_features_dim=input_dim
                        ),
                        # attention_module(
                        #     input_dim,
                        #     num_heads=num_attention_heads,
                        #     head_dim=attention_head_dim,
                        # ),
                        downsample_module(input_dim, output_dim),
                    ]
                )
            )

        # Define Middle Block
        middle_dim = stagewise_dimensions[-1]
        self.middle_block_1 = resnet_module(
            middle_dim, middle_dim, condition_features_dim=middle_dim
        )
        # self.middle_attention = full_attention_module(
        #     middle_dim,
        #     num_heads=stagewise_num_attention_heads[-1],
        #     head_dim=stagewise_attention_head_dim[-1],
        # )
        self.middle_block_2 = resnet_module(
            middle_dim, middle_dim, condition_features_dim=middle_dim
        )

        # self.final_dim = final_dim if exists(final_dim) else num_channels

        # if not self.only_auxiliary:
        #     # Define Upsampling Layers
        #     for index, (input_dim, output_dim) in enumerate(
        #         reversed(stagewise_input_to_output_dims)
        #     ):
        #         is_last = index == (num_stages - 1)

        #         # attention_module = (
        #         #     full_attention_module if use_full_attention else LinearAttention
        #         # )
        #         upsample_module = (
        #             Upsample
        #             if not is_last
        #             else partial(nn.Conv2d, kernel_size=3, padding=1)
        #         )

        #         self.up_layers.append(
        #             nn.ModuleList(
        #                 [
        #                     resnet_module_without_fdn(
        #                         output_dim + input_dim, output_dim
        #                     ),
        #                     resnet_module_without_fdn(
        #                         output_dim + input_dim, output_dim
        #                     ),
        #                     # attention_module(
        #                     #     output_dim,
        #                     #     num_heads=num_attention_heads,
        #                     #     head_dim=attention_head_dim,
        #                     # ),
        #                     upsample_module(output_dim, input_dim),
        #                 ]
        #             )
        #         )

        #     # Define Output Layer
        #     self.final_resnet_block = resnet_module_without_fdn(
        #         input_dim * 2, input_dim
        #     )
        #     self.final_convolution = nn.Conv2d(input_dim, self.final_dim, 1)

        # # Define Auxiliary Range Predictor
        # if not self.disable_auxiliary:
        #     self.auxiliary_range_predictor = AuxiliaryRangePredictor(
        #         input_dim=middle_dim,
        #         middle_height=image_height // (2 ** (num_stages - 1)),
        #         middle_width=image_width // (2 ** (num_stages - 1)),
        #         output_dim=self.final_dim * 2,
        #         hidden_dim=range_prediction_hidden_dim,
        #         num_layers=range_prediction_num_layers,
        #     )

    @property
    def max_resolution(self):
        return 2 ** (self.num_stages - 1)

    # def forward(self, x: Tensor, time: Tensor, x_auxiliary_condition: Tensor, x_self_condition: Optional[Tensor] = None):
    def forward(
        self,
        x: Tensor,
        x_auxiliary_condition: Tensor,
        x_self_condition: Optional[Tensor] = None,
    ):
        assert all(
            [divisible_by(dim, self.max_resolution) for dim in x.shape[-2:]]
        ), f"width and height {x.shape[-2:]} must be divisible by {self.max_resolution}"

        if exists(self.num_condition_channels):
            x_self_condition = (
                x_self_condition if exists(x_self_condition) else torch.zeros_like(x)
            )
            x = torch.cat(
                (x, x_self_condition), dim=1
            )  # concat along the channel dimension

        x = self.initial_convolution(x)

        residual = x.clone()

        # time_embedding = self.time_embedding_mlp(time)

        auxiliary_condition_features = self.condition_feature_extractor(
            x_auxiliary_condition
        )

        hidden_states = []

        for index, (
            block_1,
            block_2,
            # attention,
            downsample,
        ) in enumerate(self.down_layers):
            x = block_1(x, auxiliary_condition_features[index])
            # x = block_1(x, auxiliary_condition_features[index], time_embedding = time_embedding)
            hidden_states.append(x)

            x = block_2(x, auxiliary_condition_features[index])
            # x = block_2(x, auxiliary_condition_features[index], time_embedding = time_embedding)
            # x = attention(x) + x
            hidden_states.append(x)

            x = downsample(x)

        x = self.middle_block_1(x, auxiliary_condition_features[-1])
        # x = self.middle_block_1(x, auxiliary_condition_features[-1], time_embedding = time_embedding)
        # x = self.middle_attention(x) + x
        x = self.middle_block_2(x, auxiliary_condition_features[-1])
        # x = self.middle_block_2(x, auxiliary_condition_features[-1], time_embedding = time_embedding)
        # if not self.disable_auxiliary:
        #     auxiliary_range_tuple = self.auxiliary_range_predictor(x)
        # else:
        #     auxiliary_range_tuple = None
        # # Take the negative of even indices to get the range
        # auxiliary_range[..., [0, 2]] = -auxiliary_range[..., [0, 2]]
        # if not self.only_auxiliary:
        #     for index, (
        #         block_1,
        #         block_2,
        #         # attention,
        #         upsample,
        #     ) in enumerate(self.up_layers):
        #         x = torch.cat(
        #             (x, hidden_states.pop()), dim=1
        #         )  # concat along the channel dimension
        #         x = block_1(
        #             x
        #             # auxiliary_condition_features[-(index + 2)],
        #         )
        #         # x = block_1(x, auxiliary_condition_features[-(index + 2)], time_embedding = time_embedding)

        #         x = torch.cat(
        #             (x, hidden_states.pop()), dim=1
        #         )  # concat along the channel dimension
        #         x = block_2(
        #             x
        #             # auxiliary_condition_features[-(index + 2)],
        #         )
        #         # x = block_2(x, auxiliary_condition_features[-(index + 2)], time_embedding = time_embedding)
        #         # x = attention(x) + x

        #         x = upsample(x)

        #     x = torch.cat((x, residual), dim=1)  # concat along the channel dimension

        #     x = self.final_resnet_block(
        #         x,
        #         # auxiliary_condition_features[0],
        #     )
        #     # x = self.final_resnet_block(x, auxiliary_condition_features[0], time_embedding = time_embedding)
        #     x = self.final_convolution(x)
        #     x = x / x.amax(dim=(-2, -1), keepdim=True)  # normalize to [-1, 1] range

        return x, hidden_states, residual


class FDNUNetDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        initial_dim: Optional[int] = None,
        final_dim: Optional[int] = None,
        num_stages: int = 4,
        num_channels: int = 3,
        num_condition_channels: Optional[int] = None,
        resnet_num_groups_for_normalization: int = 8,
    ):
        super().__init__()

        # Define Dimensions
        self.num_stages = num_stages

        stagewise_dim_multipliers = tuple([2**i for i in range(num_stages)])

        self.num_channels = num_channels

        self.num_condition_channels = num_condition_channels

        initial_dim = initial_dim if exists(initial_dim) else input_dim

        stagewise_dimensions = [
            initial_dim,
            *map(lambda multiplier: input_dim * multiplier, stagewise_dim_multipliers),
        ]
        assert (
            len(stagewise_dimensions) == num_stages + 1
        ), "stagewise dimensions must be equal to number of stages + 1"

        stagewise_input_to_output_dims = list(
            zip(stagewise_dimensions[:-1], stagewise_dimensions[1:])
        )
        assert (
            len(stagewise_input_to_output_dims) == num_stages
        ), "stagewise input to output dimensions must be equal to number of stages"

        # # Define Time Embedding
        # time_embedding_dim = input_dim * 4

        # self.time_embedding_mlp = nn.Sequential(
        #     SinusoidalPosEmb(input_dim, theta=positional_embedding_theta),
        #     nn.Linear(input_dim, time_embedding_dim),
        #     nn.GELU(),
        #     nn.Linear(time_embedding_dim, time_embedding_dim),
        # )

        # Define Resnet
        # resnet_module = partial(ConditionedResnetBlock, num_groups_for_normalization = resnet_num_groups_for_normalization, time_embedding_dim = time_embedding_dim)
        resnet_module_without_fdn = partial(
            ConditionedResnetBlock,
            use_fdn=False,
            num_groups_for_normalization=resnet_num_groups_for_normalization,
        )

        self.final_dim = final_dim if exists(final_dim) else num_channels

        self.up_layers = nn.ModuleList([])

        for index, (input_dim, output_dim) in enumerate(
            reversed(stagewise_input_to_output_dims)
        ):
            is_last = index == (num_stages - 1)

            # attention_module = (
            #     full_attention_module if use_full_attention else LinearAttention
            # )
            upsample_module = (
                Upsample
                if not is_last
                else partial(nn.Conv2d, kernel_size=3, padding=1)
            )

            self.up_layers.append(
                nn.ModuleList(
                    [
                        resnet_module_without_fdn(output_dim + input_dim, output_dim),
                        resnet_module_without_fdn(output_dim + input_dim, output_dim),
                        # attention_module(
                        #     output_dim,
                        #     num_heads=num_attention_heads,
                        #     head_dim=attention_head_dim,
                        # ),
                        upsample_module(output_dim, input_dim),
                    ]
                )
            )

        # Define Output Layer
        self.final_resnet_block = resnet_module_without_fdn(input_dim * 2, input_dim)
        self.final_convolution = nn.Conv2d(input_dim, self.final_dim, 1)

    def forward(self, x: Tensor, hidden_states: List[Tensor], residual: Tensor):
        for index, (
            block_1,
            block_2,
            # attention,
            upsample,
        ) in enumerate(self.up_layers):
            x = torch.cat(
                (x, hidden_states.pop()), dim=1
            )  # concat along the channel dimension
            x = block_1(
                x
                # auxiliary_condition_features[-(index + 2)],
            )
            # x = block_1(x, auxiliary_condition_features[-(index + 2)], time_embedding = time_embedding)

            x = torch.cat(
                (x, hidden_states.pop()), dim=1
            )  # concat along the channel dimension
            x = block_2(
                x
                # auxiliary_condition_features[-(index + 2)],
            )
            # x = block_2(x, auxiliary_condition_features[-(index + 2)], time_embedding = time_embedding)
            # x = attention(x) + x

            x = upsample(x)

        x = torch.cat((x, residual), dim=1)  # concat along the channel dimension

        x = self.final_resnet_block(
            x,
            # auxiliary_condition_features[0],
        )
        # x = self.final_resnet_block(x, auxiliary_condition_features[0], time_embedding = time_embedding)
        x = self.final_convolution(x)
        x = x / x.amax(dim=(-2, -1), keepdim=True)  # normalize to [-1, 1] range

        return x


class FDNUNetAuxDecoder(nn.Module):
    def __init__(
        self,
        middle_dim: int,
        final_dim: int,
        image_height: int,
        image_width: int,
        range_prediction_hidden_dim: int = 256,
        range_prediction_num_layers: int = 3,
        num_stages: int = 4,
    ):
        super().__init__()
        self.auxiliary_range_predictor = AuxiliaryRangePredictor(
            input_dim=middle_dim,
            middle_height=image_height // (2 ** (num_stages - 1)),
            middle_width=image_width // (2 ** (num_stages - 1)),
            output_dim=final_dim * 2,
            hidden_dim=range_prediction_hidden_dim,
            num_layers=range_prediction_num_layers,
        )

    def forward(self, x: Tensor):
        auxiliary_range_tuple = self.auxiliary_range_predictor(x)
        return auxiliary_range_tuple


def create_models(
    input_dim: int,
    image_height: int,
    image_width: int,
    range_prediction_hidden_dim: int = 256,
    range_prediction_num_layers: int = 3,
    initial_dim: Optional[int] = None,
    final_dim: Optional[int] = None,
    num_stages: int = 4,
    num_channels: int = 3,
    num_auxiliary_condition_channels: int = 3,
    num_condition_channels: Optional[int] = None,
    resnet_num_groups_for_normalization: int = 8,
):
    stagewise_dim_multipliers = tuple([2**i for i in range(num_stages)])

    initial_dim = initial_dim if exists(initial_dim) else input_dim

    stagewise_dimensions = [
        initial_dim,
        *map(lambda multiplier: input_dim * multiplier, stagewise_dim_multipliers),
    ]
    assert (
        len(stagewise_dimensions) == num_stages + 1
    ), "stagewise dimensions must be equal to number of stages + 1"

    stagewise_input_to_output_dims = list(
        zip(stagewise_dimensions[:-1], stagewise_dimensions[1:])
    )
    assert (
        len(stagewise_input_to_output_dims) == num_stages
    ), "stagewise input to output dimensions must be equal to number of stages"

    encoder = FDNUNetEncoder(
        input_dim=input_dim,
        initial_dim=initial_dim,
        num_stages=num_stages,
        num_channels=num_channels,
        num_auxiliary_condition_channels=num_auxiliary_condition_channels,
        num_condition_channels=num_condition_channels,
        resnet_num_groups_for_normalization=resnet_num_groups_for_normalization,
    )

    decoder = FDNUNetDecoder(
        input_dim=input_dim,
        initial_dim=initial_dim,
        final_dim=final_dim,
        num_stages=num_stages,
        num_channels=num_channels,
        num_condition_channels=num_condition_channels,
        resnet_num_groups_for_normalization=resnet_num_groups_for_normalization,
    )

    auxiliary = FDNUNetAuxDecoder(
        middle_dim=stagewise_dimensions[-1],
        final_dim=final_dim if exists(final_dim) else num_channels,
        image_height=image_height,
        image_width=image_width,
        range_prediction_hidden_dim=range_prediction_hidden_dim,
        range_prediction_num_layers=range_prediction_num_layers,
        num_stages=num_stages,
    )

    return encoder, decoder, auxiliary


# class FDNUNetWithAux(nn.Module):
#     def __init__(
#         self,
#         encoder: nn.Module,
#         decoder: nn.Module,
#         auxiliary: nn.Module,
#         disable_auxiliary=False,
#         only_auxiliary=False,
#         # positional_embedding_theta: int = 10000,
#         # attention_head_dim: int = 32,
#         # num_attention_heads: int = 4,
#         # use_full_attention: bool = False,  # defaults to full attention only for inner most layer
#         # use_flash_attention: bool = False,
#     ):
#         super().__init__()

#         self.disable_auxiliary = disable_auxiliary

#         self.only_auxiliary = only_auxiliary

#         assert not (
#             self.disable_auxiliary and self.only_auxiliary
#         ), "Cannot disable and only use auxiliary"

#         self.encoder = encoder

#         if not self.only_auxiliary:
#             self.decoder = decoder

#         if not self.disable_auxiliary:
#             self.auxiliary = auxiliary

#     def forward(
#         self,
#         x: Tensor,
#         x_auxiliary_condition: Tensor,
#         x_self_condition: Optional[Tensor] = None,
#     ):
#         x, hidden_states, residual = self.encoder.forward(
#             x, x_auxiliary_condition, x_self_condition
#         )
#         if not self.disable_auxiliary:
#             auxiliary_range_tuple = self.auxiliary.forward(x)
#         if not self.only_auxiliary:
#             x = self.decoder.forward(x, hidden_states, residual)
#         return x, auxiliary_range_tuple
