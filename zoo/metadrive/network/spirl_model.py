from torch import nn
from typing import Union, Dict, Optional, List
from easydict import EasyDict
import torch

from ding.utils import SequenceType, squeeze
from ding.model.template import QAC
from ding.model.common import RegressionHead, ReparameterizationHead, FCEncoder
from typing import Tuple, Optional
from ding.torch_utils import ResBlock, Flatten
class ConvEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            obs_shape: Tuple,
            hidden_size_list: Tuple = [32, 64, 128],
            kernel_size: Tuple = [3, 3, 3],
            stride: Tuple = [2, 2, 2],
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        r"""
        Overview:
            Init the Convolution Encoder according to arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, some ``output size``
            - hidden_size_list (:obj:`SequenceType`): The collection of ``hidden_size``
            - activation (:obj:`nn.Module`):
                The type of activation to use in the conv ``layers`` and ``ResBlock``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`str`):
                The type of normalization to use, see ``ding.torch_utils.ResBlock`` for more details
        """
        super(ConvEncoder, self).__init__()
        self._obs_shape = obs_shape
        self._activation = activation
        self._hidden_size_list = hidden_size_list

        layers = []
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i]))
            if self._activation is not None:
                layers.append(self._activation)
            input_size = hidden_size_list[i]
        assert len(set(hidden_size_list[3:-1])) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(3, len(self._hidden_size_list) - 1):
            layers.append(ResBlock(self._hidden_size_list[i], activation=self.act, norm_type=norm_type))
        layers.append(Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        r"""
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Arguments:
            - x (:obj:`torch.Tensor`): Encoded Tensor after ``self.main``
        Returns:
            - outputs (:obj:`torch.Tensor`): Size int, also number of in-feature
        """
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return embedding tensor of the env observation
        Arguments:
            - x (:obj:`torch.Tensor`): Env raw observation
        Returns:
            - outputs (:obj:`torch.Tensor`): Embedding tensor
        """
        x = self.main(x)
        x = self.mid(x)
        return x


class SpirlEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType, EasyDict],
            encoder_hidden_size_list: SequenceType = [64],
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        super(SpirlEncoder, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(obs_shape)
            )
        self.action_shape = action_shape
        self.actor = nn.Sequential(
            encoder_cls(obs_shape, encoder_hidden_size_list, activation=None, norm_type=norm_type), activation,
            RegressionHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                final_tanh=True,
                activation=activation,
                norm_type=norm_type
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.actor(x)
        return x['pred']