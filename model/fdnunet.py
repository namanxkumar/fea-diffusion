from .attentionutils import Attend

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from functools import partial
import math

from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from typing import Optional, cast, List, Tuple

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
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=downsample_factor, p2=downsample_factor),
            nn.Conv2d(input_dim * (downsample_factor ** 2), output_dim, 1)
        )

    def forward(self, x: Tensor):
        return self.downsample(x)
    
class Upsample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        upsample_factor = 2
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
            nn.Conv2d(input_dim, output_dim, 3, padding = 1) # [gen] Convolution layer after the upsample is to smooth out the upsampled image
        )

    def forward(self, x: Tensor):
        return self.upsample(x)
    
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_to_normalize = 1 # Normalize over the channel dimension
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor):
        # RMS Norm is given by:
        # [x / sqrt(E[x^2])] * g
        # where g is a learnable parameter
        # and E[x^2] is the mean of the square of x
        # sqrt(E[x^2]) = sqrt(Sum(x^2) / N)
        # so [x / sqrt(E[x^2])] = [x * sqrt(N) / sqrt(Sum(x^2))]
        # F.normalize gives us [x / sqrt(Sum(x^2))], which is multiplied by sqrt(N) to give us the desired result
        return F.normalize(x, dim = self.dim_to_normalize) * (x.shape[self.dim_to_normalize] ** 0.5)  * self.g
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, embedding_dim: int, theta: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta

    def forward(self, positions: Tensor):
        # Similar to pytorch's positional embedding, which uses log form for efficiency, explained here: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6

        device = positions.device
        embedding = math.log(self.theta) / ((self.embedding_dim//2) - 1)
        embedding = torch.exp(torch.arange((self.embedding_dim//2), device=device) * -embedding)
        embedding = positions[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding

class FeatureDenormalization(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, num_groups_for_normalization: int = 8):
        super().__init__()

        self.normalize = nn.GroupNorm(num_groups_for_normalization, input_dim, affine=False)
        self.scale_convolution = nn.Conv2d(feature_dim, input_dim, kernel_size=3, padding=1)
        self.shift_convolution = nn.Conv2d(feature_dim, input_dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, features: Tensor):
        assert features.size()[2:] == x.size()[2:], 'features must have the same spatial dimensions as x'
        x = self.normalize(x)
        scale = self.scale_convolution(features)
        shift = self.shift_convolution(features)
        x = x * (scale + 1) + shift
        return self.activation(x)
    
class ConditionFeatureExtractor(nn.Module):
    def __init__(self, num_condition_channels: int, stagewise_dimensions: List[int]):
        super().__init__()

        stagewise_input_to_output_dims = list(zip(stagewise_dimensions[:-1], stagewise_dimensions[1:]))

        self.pre_extractors = nn.Sequential(
            nn.Conv2d(num_condition_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, stagewise_input_to_output_dims[0][0], kernel_size=3, padding=1),
                nn.ReLU(),
            ),
        ])

        for index, (input_dim, output_dim) in enumerate(stagewise_input_to_output_dims):
            module = nn.Sequential(
                (nn.Conv2d(input_dim, output_dim, kernel_size = 3, padding = 1) if (index == len(stagewise_input_to_output_dims) - 1) else Downsample(input_dim, output_dim)),
                nn.ReLU(),
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
        self.project = nn.Conv2d(input_dim, output_dim, 3, padding = 1)
        self.activate = nn.ReLU()

    def forward(self, x: Tensor, scale_shift = None):
        x = self.project(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activate(x)
        return x
    
class ConditionedResnetBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, condition_features_dim: int, *, time_embedding_dim: Optional[int] = None, num_groups_for_normalization: int = 8):
        super().__init__()

        self.time_embedding_to_scale_shift = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_embedding_dim, output_dim * 2)
        ) if exists(time_embedding_dim) else None

        self.conditional_block1 = FeatureDenormalization(input_dim, condition_features_dim, num_groups_for_normalization = num_groups_for_normalization)
        self.block1 = LeanResnetSubBlock(input_dim, output_dim)
        self.conditional_block2 = FeatureDenormalization(output_dim, condition_features_dim, num_groups_for_normalization = num_groups_for_normalization)
        self.block2 = LeanResnetSubBlock(output_dim, output_dim)
        
        self.residual_convolution = nn.Conv2d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, condition_features: torch.Tensor, time_embedding: Optional[Tensor] = None):
        h = self.conditional_block1(x, condition_features)

        scale_shift = None
        if exists(self.time_embedding_to_scale_shift) and exists(time_embedding):
            time_embedding = self.time_embedding_to_scale_shift(time_embedding)
            time_embedding = rearrange(time_embedding, 'b c -> b c 1 1')
            scale_shift = time_embedding.chunk(2, dim = 1) # split into scale and shift each of dimension b c/2 1 1

        h = self.block1(h, scale_shift = scale_shift)
        h = self.conditional_block2(h, condition_features)
        h = self.block2(h)

        return h + self.residual_convolution(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        head_dim: int = 32,
        num_memory_key_value: int = 4
    ):
        super().__init__()
        self.scaling_factor = head_dim ** -0.5
        self.num_heads = num_heads
        hidden_dim = num_heads * head_dim

        self.normalize = RMSNorm(input_dim)

        self.memory_key_value = nn.Parameter(torch.randn(2, num_heads, head_dim, num_memory_key_value)) # 2 for keys and values, num_mem_kv is the number of memory slots
        self.to_query_key_value = nn.Conv2d(input_dim, hidden_dim * 3, 1, bias = False) 

        self.to_output = nn.Sequential(
            nn.Conv2d(hidden_dim, input_dim, 1),
            RMSNorm(input_dim)
        )

    def forward(self, x: Tensor):
        batch_size, _, image_height, image_width = x.shape

        x = self.normalize(x)

        query_key_value: Tensor = cast(Tensor, self.to_query_key_value(x)).chunk(3, dim = 1) # split into q, k, v each of dimension [b, hidden_dim, h, w]

        query, key, value = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.num_heads), query_key_value) # split into heads, c is dim_head

        memory_key, memory_value = map(lambda t: repeat(t, 'h c n -> b h c n', b = batch_size), self.memory_key_value) # repeat memory keys and values across batch dimension
        key, value = map(partial(torch.cat, dim = -1), ((memory_key, key), (memory_value, value))) # concat memory keys and values with the keys and values from the input, shape is b, heads, dim_head, num_mem_kv + w * h

        query = query.softmax(dim = -2) # softmax over the dim_head (c) dimension
        key = key.softmax(dim = -1) # softmax over the (num_mem_kv + w * h) dimension

        query = query * self.scaling_factor # scale q by the sqrt of the dim_head

        context = torch.einsum('b h d n, b h e n -> b h d e', key, value) # context between memory keys and values, shape is b, heads, dim_head, dim_head. d, e are dim_head

        output = torch.einsum('b h d e, b h d n -> b h e n', context, query) # output is query multiplied by the context, shape is b, heads, dim_head, w * h

        output = rearrange(output, 'b h c (x y) -> b (h c) x y', h = self.num_heads, x = image_height, y = image_width) # concat heads and dim_head dimensions

        return self.to_output(output)

class FullAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        head_dim: int = 32,
        num_memory_key_value: int = 4,
        use_flash_attention: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        hidden_dim = num_heads * head_dim 

        self.normalize = RMSNorm(input_dim)
        self.attend = Attend(flash = use_flash_attention)

        self.memory_key_value = nn.Parameter(torch.randn(2, num_heads, num_memory_key_value, head_dim))
        self.to_query_key_value = nn.Conv2d(input_dim, hidden_dim * 3, 1, bias = False)
        self.to_output = nn.Conv2d(hidden_dim, input_dim, 1)

    def forward(self, x: Tensor):
        batch_size, _, image_height, image_width = x.shape

        x = self.normalize(x)

        query_key_value = cast(Tensor, self.to_query_key_value(x)).chunk(3, dim = 1)
        query, key, value = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.num_heads), query_key_value)

        memory_key, memory_value = map(lambda t: repeat(t, 'h n d -> b h n d', b = batch_size), self.memory_key_value)
        key, value = map(partial(torch.cat, dim = -2), ((memory_key, key), (memory_value, value)))

        output = self.attend(query, key, value)

        output = rearrange(output, 'b h (x y) d -> b (h d) x y', x = image_height, y = image_width)

        return self.to_output(output)

class FDNUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        initial_dim: Optional[int] = None,
        final_dim: Optional[int] = None,
        num_stages: int = 4,
        num_channels: int = 3,
        num_auxiliary_condition_channels: int = 3,
        num_condition_channels: Optional[int] = None,
        resnet_num_groups_for_normalization: int = 8,
        positional_embedding_theta: int = 10000,
        attention_head_dim: int = 32,
        num_attention_heads: int = 4,
        use_full_attention: bool = False,    # defaults to full attention only for inner most layer
        use_flash_attention: bool = False
    ):
        super().__init__()

        # Define Dimensions
        self.num_stages = num_stages

        stagewise_dim_multipliers = tuple([2 ** i for i in range(num_stages)])

        self.num_channels = num_channels

        self.num_condition_channels = num_condition_channels

        if exists(num_condition_channels):
            input_channels = num_channels + num_condition_channels
        else:
            input_channels = num_channels
        
        initial_dim = initial_dim if exists(initial_dim) else input_dim

        stagewise_dimensions = [initial_dim, *map(lambda multiplier: input_dim * multiplier, stagewise_dim_multipliers)]
        assert len(stagewise_dimensions) == num_stages + 1, 'stagewise dimensions must be equal to number of stages + 1'

        stagewise_input_to_output_dims = list(zip(stagewise_dimensions[:-1], stagewise_dimensions[1:]))
        assert len(stagewise_input_to_output_dims) == num_stages, 'stagewise input to output dimensions must be equal to number of stages'
        
        # Define Time Embedding
        time_embedding_dim = input_dim * 4

        self.time_embedding_mlp = nn.Sequential(
            SinusoidalPosEmb(input_dim, theta = positional_embedding_theta),
            nn.Linear(input_dim, time_embedding_dim),
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Define Condition Feature Extractor
        self.condition_feature_extractor = ConditionFeatureExtractor(num_auxiliary_condition_channels, stagewise_dimensions)

        # Define Resnet
        # resnet_module = partial(ConditionedResnetBlock, num_groups_for_normalization = resnet_num_groups_for_normalization, time_embedding_dim = time_embedding_dim)
        resnet_module = partial(ConditionedResnetBlock, num_groups_for_normalization = resnet_num_groups_for_normalization)

        # Define Attention
        if not use_full_attention:
            stagewise_use_full_attention = (*((False,) * (num_stages - 1)), True)
        else:
            stagewise_use_full_attention = (True,) * num_stages

        stagewise_num_attention_heads = (num_attention_heads,) * num_stages
        stagewise_attention_head_dim = (attention_head_dim,) * num_stages

        assert len(stagewise_use_full_attention) == len(stagewise_num_attention_heads) == len(stagewise_attention_head_dim) == num_stages, 'number of stages must be equal for attention parameters'

        full_attention_module = partial(FullAttention, use_flash_attention = use_flash_attention)

        ## Build Layers
        self.initial_convolution = nn.Conv2d(input_channels, initial_dim, 3, padding = 1)

        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])

        # Define Downsampling Layers
        for index, ((input_dim, output_dim), use_full_attention, num_attention_heads, attention_head_dim) in enumerate(zip(stagewise_input_to_output_dims, stagewise_use_full_attention, stagewise_num_attention_heads, stagewise_attention_head_dim)):
            is_last = index == (num_stages - 1)

            attention_module = full_attention_module if use_full_attention else LinearAttention
            downsample_module = Downsample if not is_last else partial(nn.Conv2d, kernel_size = 3, padding = 1)

            self.down_layers.append(nn.ModuleList([
                resnet_module(input_dim, input_dim, input_dim),
                resnet_module(input_dim, input_dim, input_dim),
                attention_module(input_dim, num_heads = num_attention_heads, head_dim = attention_head_dim),
                downsample_module(input_dim, output_dim)
            ]))

        # Define Middle Block
        middle_dim = stagewise_dimensions[-1]
        self.middle_block_1 = resnet_module(middle_dim, middle_dim, middle_dim)
        self.middle_attention = full_attention_module(middle_dim, num_heads = stagewise_num_attention_heads[-1], head_dim = stagewise_attention_head_dim[-1])
        self.middle_block_2 = resnet_module(middle_dim, middle_dim, middle_dim)

        # Define Upsampling Layers
        for index, ((input_dim, output_dim), use_full_attention, num_attention_heads, attention_head_dim) in enumerate(zip(*map(reversed, (stagewise_input_to_output_dims, stagewise_use_full_attention, stagewise_num_attention_heads, stagewise_attention_head_dim)))):
            is_last = index == (num_stages - 1)

            attention_module = full_attention_module if use_full_attention else LinearAttention
            upsample_module = Upsample if not is_last else partial(nn.Conv2d, kernel_size = 3, padding = 1)

            self.up_layers.append(nn.ModuleList([
                resnet_module(output_dim + input_dim, output_dim, input_dim),
                resnet_module(output_dim + input_dim, output_dim, input_dim),
                attention_module(output_dim, num_heads = num_attention_heads, head_dim = attention_head_dim),
                upsample_module(output_dim, input_dim)
            ]))

        # Define Output Layer
        self.final_dim = final_dim if exists(final_dim) else num_channels

        self.final_resnet_block = resnet_module(input_dim*2, input_dim, input_dim)
        self.final_convolution = nn.Conv2d(input_dim, self.final_dim, 1)

    @property
    def max_resolution(self):
        return 2 ** (self.num_stages - 1)
    
    # def forward(self, x: Tensor, time: Tensor, x_auxiliary_condition: Tensor, x_self_condition: Optional[Tensor] = None):
    def forward(self, x: Tensor, x_auxiliary_condition: Tensor, x_self_condition: Optional[Tensor] = None):
        assert all([divisible_by(dim, self.max_resolution) for dim in x.shape[-2:]]), f'width and height {x.shape[-2:]} must be divisible by {self.max_resolution}'
        
        if exists(self.num_condition_channels):
            x_self_condition = x_self_condition if exists(x_self_condition) else torch.zeros_like(x)
            x = torch.cat((x, x_self_condition), dim = 1) # concat along the channel dimension

        x = self.initial_convolution(x)

        residual = x.clone()

        # time_embedding = self.time_embedding_mlp(time)

        auxiliary_condition_features = self.condition_feature_extractor(x_auxiliary_condition)

        hidden_states = []

        for index, (block_1, block_2, attention, downsample) in enumerate(self.down_layers):
            x = block_1(x, auxiliary_condition_features[index])
            # x = block_1(x, auxiliary_condition_features[index], time_embedding = time_embedding)
            hidden_states.append(x)

            x = block_2(x, auxiliary_condition_features[index])
            # x = block_2(x, auxiliary_condition_features[index], time_embedding = time_embedding)
            x = attention(x) + x
            hidden_states.append(x)

            x = downsample(x)

        x = self.middle_block_1(x, auxiliary_condition_features[-1])
        # x = self.middle_block_1(x, auxiliary_condition_features[-1], time_embedding = time_embedding)
        x = self.middle_attention(x) + x
        x = self.middle_block_2(x, auxiliary_condition_features[-1])
        # x = self.middle_block_2(x, auxiliary_condition_features[-1], time_embedding = time_embedding)

        for index, (block_1, block_2, attention, upsample) in enumerate(self.up_layers):
            x = torch.cat((x, hidden_states.pop()), dim = 1) # concat along the channel dimension
            x = block_1(x, auxiliary_condition_features[-(index + 2)])
            # x = block_1(x, auxiliary_condition_features[-(index + 2)], time_embedding = time_embedding)
            
            x = torch.cat((x, hidden_states.pop()), dim = 1) # concat along the channel dimension
            x = block_2(x, auxiliary_condition_features[-(index + 2)])
            # x = block_2(x, auxiliary_condition_features[-(index + 2)], time_embedding = time_embedding)
            x = attention(x) + x

            x = upsample(x)

        x = torch.cat((x, residual), dim = 1) # concat along the channel dimension

        x = self.final_resnet_block(x, auxiliary_condition_features[0])
        # x = self.final_resnet_block(x, auxiliary_condition_features[0], time_embedding = time_embedding)
        x = self.final_convolution(x)
        x = x / x.amax(dim = (-2, -1), keepdim = True) # normalize to [-1, 1] range
        return x