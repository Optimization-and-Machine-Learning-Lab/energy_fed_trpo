from __future__ import annotations

import warnings

from numbers import Number
from typing import Callable, List, Sequence, Tuple, Type, Optional, Union

import torch
from torch import nn

from torchrl._utils import prod
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.models import MultiAgentNetBase
from torchrl.modules.models.models import _ExecutableLayer, _iter_maybe_over_single
from torchrl.modules.models.utils import (
    create_on_device,
    LazyMapping,
)

class MLP(nn.Sequential):

    """A multi-layer perceptron.

    If MLP receives more than one input, it concatenates them all along the last dimension before passing the
    resulting tensor through the network. This is aimed at allowing for a seamless interface with calls of the type of

        >>> model(state, action)  # compute state-action value

    In the future, this feature may be moved to the ProbabilisticTDModule, though it would require it to handle
    different cases (vectors, images, ...)

    Args:
        in_features (int, optional): number of input features;
        in_features_names (List[str], optional): names of input features to group them into related topics for a more expressive network.
        out_features (int, torch.Size or equivalent): number of output
            features. If iterable of integers, the output is reshaped to the
            desired shape.
        depth (int, optional): depth of the network. A depth of 0 will produce
            a single linear layer network with the desired input and output size.
            A length of 1 will create 2 linear layers etc. If no depth is indicated,
            the depth information should be contained in the ``num_cells``
            argument (see below). If ``num_cells`` is an iterable and depth is
            indicated, both should match: ``len(num_cells)`` must be equal to
            ``depth``.
        num_cells (int or sequence of int, optional): number of cells of every
            layer in between the input and output. If an integer is provided,
            every layer will have the same number of cells. If an iterable is provided,
            the linear layers ``out_features`` will match the content of
            ``num_cells``. Defaults to ``32``;
        activation_class (Type[nn.Module] or callable, optional): activation
            class or constructor to be used.
            Defaults to :class:`~torch.nn.Tanh`.
        activation_kwargs (dict or list of dicts, optional): kwargs to be used
            with the activation class. Aslo accepts a list of kwargs of length
            ``depth + int(activate_last_layer)``.
        norm_class (Type or callable, optional): normalization class or
            constructor, if any.
        norm_kwargs (dict or list of dicts, optional): kwargs to be used with
            the normalization layers. Aslo accepts a list of kwargs of length
            ``depth + int(activate_last_layer)``.
        dropout (float, optional): dropout probability. Defaults to ``None`` (no
            dropout);
        bias_last_layer (bool): if ``True``, the last Linear layer will have a bias parameter.
            default: True;
        single_bias_last_layer (bool): if ``True``, the last dimension of the bias of the last layer will be a singleton
            dimension.
            default: True;
        layer_class (Type[nn.Module] or callable, optional): class to be used
            for the linear layers;
        layer_kwargs (dict or list of dicts, optional): kwargs for the linear
            layers. Aslo accepts a list of kwargs of length ``depth + 1``.
        activate_last_layer (bool): whether the MLP output should be activated. This is useful when the MLP output
            is used as the input for another module.
            default: False.
        device (torch.device, optional): device to create the module on.

    Examples:
        >>> # All of the following examples provide valid, working MLPs
        >>> mlp = MLP(in_features=3, out_features=6, depth=0) # MLP consisting of a single 3 x 6 linear layer
        >>> print(mlp)
        MLP(
          (0): Linear(in_features=3, out_features=6, bias=True)
        )
        >>> mlp = MLP(in_features=3, out_features=6, depth=4, num_cells=32)
        >>> print(mlp)
        MLP(
          (0): Linear(in_features=3, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): Tanh()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): Tanh()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): Tanh()
          (8): Linear(in_features=32, out_features=6, bias=True)
        )
        >>> mlp = MLP(out_features=6, depth=4, num_cells=32)  # LazyLinear for the first layer
        >>> print(mlp)
        MLP(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): Tanh()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): Tanh()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): Tanh()
          (8): Linear(in_features=32, out_features=6, bias=True)
        )
        >>> mlp = MLP(out_features=6, num_cells=[32, 33, 34, 35])  # defines the depth by the num_cells arg
        >>> print(mlp)
        MLP(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=33, bias=True)
          (3): Tanh()
          (4): Linear(in_features=33, out_features=34, bias=True)
          (5): Tanh()
          (6): Linear(in_features=34, out_features=35, bias=True)
          (7): Tanh()
          (8): Linear(in_features=35, out_features=6, bias=True)
        )
        >>> mlp = MLP(out_features=(6, 7), num_cells=[32, 33, 34, 35])  # returns a view of the output tensor with shape [*, 6, 7]
        >>> print(mlp)
        MLP(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): Tanh()
          (2): Linear(in_features=32, out_features=33, bias=True)
          (3): Tanh()
          (4): Linear(in_features=33, out_features=34, bias=True)
          (5): Tanh()
          (6): Linear(in_features=34, out_features=35, bias=True)
          (7): Tanh()
          (8): Linear(in_features=35, out_features=42, bias=True)
        )
        >>> from torchrl.modules import NoisyLinear
        >>> mlp = MLP(out_features=(6, 7), num_cells=[32, 33, 34, 35], layer_class=NoisyLinear)  # uses NoisyLinear layers
        >>> print(mlp)
        MLP(
          (0): NoisyLazyLinear(in_features=0, out_features=32, bias=False)
          (1): Tanh()
          (2): NoisyLinear(in_features=32, out_features=33, bias=True)
          (3): Tanh()
          (4): NoisyLinear(in_features=33, out_features=34, bias=True)
          (5): Tanh()
          (6): NoisyLinear(in_features=34, out_features=35, bias=True)
          (7): Tanh()
          (8): NoisyLinear(in_features=35, out_features=42, bias=True)
        )

    """

    def __init__(
        self,
        in_features: int | None = None,
        in_features_names: List[str] | None = None,
        out_features: int | torch.Size = None,
        depth: int | None = None,
        num_cells: Sequence[int] | int | None = None,
        activation_class: Type[nn.Module] | Callable = nn.Tanh,
        activation_kwargs: dict | List[dict] | None = None,
        norm_class: Type[nn.Module] | Callable | None = None,
        norm_kwargs: dict | List[dict] | None = None,
        dropout: float | None = None,
        bias_last_layer: bool = True,
        single_bias_last_layer: bool = False,
        layer_class: Type[nn.Module] | Callable = nn.Linear,
        layer_kwargs: dict | None = None,
        activate_last_layer: bool = False,
        device: DEVICE_TYPING | None = None,
    ):
        if out_features is None:
            raise ValueError("out_features must be specified for MLP.")

        if num_cells is None:
            warnings.warn(
                "The current behavior of MLP when not providing `num_cells` is that the number of cells is "
                "set to [default_num_cells] * depth, where `depth=3` by default and `default_num_cells=0`. "
                "From v0.7, this behavior will switch and `depth=0` will be used. "
                "To silence tis message, indicate what number of cells you desire.",
                category=DeprecationWarning,
            )
            default_num_cells = 32
            if depth is None:
                num_cells = [default_num_cells] * 3
                depth = 3
            else:
                num_cells = [default_num_cells] * depth

        self.in_features = in_features
        self.in_features_names = in_features_names

        _out_features_num = out_features
        if not isinstance(out_features, Number):
            _out_features_num = prod(out_features)
        self.out_features = out_features
        self._out_features_num = _out_features_num
        self.activation_class = activation_class
        self.norm_class = norm_class
        self.dropout = dropout
        self.bias_last_layer = bias_last_layer
        self.single_bias_last_layer = single_bias_last_layer
        self.layer_class = layer_class

        self.activation_kwargs = activation_kwargs
        self.norm_kwargs = norm_kwargs
        self.layer_kwargs = layer_kwargs

        self.activate_last_layer = activate_last_layer
        if single_bias_last_layer:
            raise NotImplementedError

        if not (isinstance(num_cells, Sequence) or depth is not None):
            raise RuntimeError(
                "If num_cells is provided as an integer, \
            depth must be provided too."
            )
        self.num_cells = (
            list(num_cells) if isinstance(num_cells, Sequence) else [num_cells] * depth
        )
        self.depth = depth if depth is not None else len(self.num_cells)
        if not (len(self.num_cells) == depth or depth is None):
            raise RuntimeError(
                "depth and num_cells length conflict, \
            consider matching or specifying a constant num_cells argument together with a a desired depth"
            )

        self._activation_kwargs_iter = _iter_maybe_over_single(
            activation_kwargs, n=self.depth + self.activate_last_layer
        )
        self._norm_kwargs_iter = _iter_maybe_over_single(
            norm_kwargs, n=self.depth + self.activate_last_layer
        )
        self._layer_kwargs_iter = _iter_maybe_over_single(
            layer_kwargs, n=self.depth + 1
        )

        groupped_features_module = self._make_grouped_features_module(device)
        layers = self._make_net(device)

        layers.pop(0) # We are replacing the input layer for our grouped features module

        layers = groupped_features_module + layers
        layers = [
            layer if isinstance(layer, nn.Module) else _ExecutableLayer(layer)
            for layer in layers
        ]
        super().__init__(*layers)

    def _make_grouped_features_module(self, device: DEVICE_TYPING | None) -> List[nn.Module]:

        # TODO: If needed, making this grouping a parameter
        # Manage grouping of features, 

        in_feature_grouping = {
            "encoded_demand": {
                "keys": ["personal_encoding", "load"],
                "indexes": [],
                "layers": []
            },
            "encoded_solar": {
                "keys": ["personal_encoding", "solar"],
                "indexes": [],
                "layers": []
            },
            "environment": {
                "keys": ["temp", "hum"],
                "indexes": [],
                "layers": []
            },
            "other": {
                "indexes": [],
                "layers": []
            }
        }

        # Pass through the in_features_names to group the indexes

        for i, name in enumerate(self.in_features_names):
            
            if any(key in name for key in in_feature_grouping["encoded_demand"]["keys"]):
                in_feature_grouping["encoded_demand"]["indexes"].append(i)
            
            if any(key in name for key in in_feature_grouping["encoded_solar"]["keys"]):
                in_feature_grouping["encoded_solar"]["indexes"].append(i)
            
            if any(key in name for key in in_feature_grouping["environment"]["keys"]):
                in_feature_grouping["environment"]["indexes"].append(i)
            
            if not any(key in name for key in in_feature_grouping["encoded_demand"]["keys"]) and \
               not any(key in name for key in in_feature_grouping["encoded_solar"]["keys"]) and \
               not any(key in name for key in in_feature_grouping["environment"]["keys"]):
                
                in_feature_grouping["other"]["indexes"].append(i)

        # Create the layers for each group

        groups_to_del = []

        for group in in_feature_grouping:

            if len(in_feature_grouping[group]["indexes"]) == 0:
                groups_to_del.append(group)
                continue

            in_features = [len(in_feature_grouping[group]["indexes"])] + self.num_cells[1:]
            out_features = self.num_cells

            group_activation_kwargs_iter = _iter_maybe_over_single(
                self.activation_kwargs, n=self.depth + 1
            )   
            group_norm_kwargs_iter = _iter_maybe_over_single(
                self.norm_kwargs, n=self.depth + 1
            )
            group_layer_kwargs_iter = _iter_maybe_over_single(
                self.layer_kwargs, n=self.depth + 1
            )

            for i, (_in, _out) in enumerate(zip(in_features, out_features)): 

                layer_kwargs = next(group_layer_kwargs_iter)
                _bias = layer_kwargs.pop(
                    "bias", self.bias_last_layer
                )

                in_feature_grouping[group]["layers"].append(
                    create_on_device(
                        self.layer_class,
                        device,
                        _in,
                        _out,
                        bias=_bias,
                        **layer_kwargs,
                    )
                )

                if i < self.depth:

                    norm_kwargs = next(group_norm_kwargs_iter)
                    activation_kwargs = next(group_activation_kwargs_iter)
                    if self.dropout is not None:
                        in_feature_grouping[group]["layers"].append(create_on_device(nn.Dropout, device, p=self.dropout))
                    if self.norm_class is not None:
                        in_feature_grouping[group]["layers"].append(
                            create_on_device(self.norm_class, device, **norm_kwargs)
                        )
                    in_feature_grouping[group]["layers"].append(
                        create_on_device(self.activation_class, device, **activation_kwargs)
                    )

        # Remove empty groups

        for group in groups_to_del:
            del in_feature_grouping[group]

        return [_GroupedByConfigLayers(in_feature_grouping=in_feature_grouping, device=device)]
            
    def _make_net(self, device: DEVICE_TYPING | None) -> List[nn.Module]:
        layers = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self._out_features_num]
        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            layer_kwargs = next(self._layer_kwargs_iter)
            _bias = layer_kwargs.pop(
                "bias", self.bias_last_layer if i == self.depth else True
            )
            if _in is not None:
                layers.append(
                    create_on_device(
                        self.layer_class,
                        device,
                        _in,
                        _out,
                        bias=_bias,
                        **layer_kwargs,
                    )
                )
            else:
                try:
                    lazy_version = LazyMapping[self.layer_class]
                except KeyError:
                    raise KeyError(
                        f"The lazy version of {self.layer_class.__name__} is not implemented yet. "
                        "Consider providing the input feature dimensions explicitely when creating an MLP module"
                    )
                layers.append(
                    create_on_device(
                        lazy_version, device, _out, bias=_bias, **layer_kwargs
                    )
                )

            if i < self.depth or self.activate_last_layer:
                norm_kwargs = next(self._norm_kwargs_iter)
                activation_kwargs = next(self._activation_kwargs_iter)
                if self.dropout is not None:
                    layers.append(create_on_device(nn.Dropout, device, p=self.dropout))
                if self.norm_class is not None:
                    layers.append(
                        create_on_device(self.norm_class, device, **norm_kwargs)
                    )
                layers.append(
                    create_on_device(self.activation_class, device, **activation_kwargs)
                )

        return layers

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)

        out = super().forward(*inputs)
        if not isinstance(self.out_features, Number):
            out = out.view(*out.shape[:-1], *self.out_features)
        return out
    
class MultiAgentMLP(MultiAgentNetBase):
    """Mult-agent MLP.

    This is an MLP that can be used in multi-agent contexts.
    For example, as a policy or as a value function.
    See `examples/multiagent` for examples.

    It expects inputs with shape (*B, n_agents, n_agent_inputs)
    It returns outputs with shape (*B, n_agents, n_agent_outputs)

    If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies).
    Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).

    If `centralized` is True, each agent will use the inputs of all agents to compute its output
    (n_agent_inputs * n_agents will be the number of inputs for one agent).
    Otherwise, each agent will only use its data as input.

    Args:
        n_agent_inputs (int or None): number of inputs for each agent. If ``None``,
            the number of inputs is lazily instantiated during the first call.
        n_agent_inputs_names (List[str] or None): names of input features to group them into related topics for a more expressive network.
        n_agent_outputs (int): number of outputs for each agent.
        n_agents (int): number of agents.

    Keyword Args:
        centralized (bool): If `centralized` is True, each agent will use the inputs of all agents to compute its output
            (n_agent_inputs * n_agents will be the number of inputs for one agent).
            Otherwise, each agent will only use its data as input.
        share_params (bool): If `share_params` is True, the same MLP will be used to make the forward pass
            for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process
            its input (heterogeneous policies).
        device (str or toech.device, optional): device to create the module on.
        depth (int, optional): depth of the network. A depth of 0 will produce a single linear layer network with the
            desired input and output size. A length of 1 will create 2 linear layers etc. If no depth is indicated,
            the depth information should be contained in the num_cells argument (see below). If num_cells is an
            iterable and depth is indicated, both should match: len(num_cells) must be equal to depth.
            default: 3.
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
            default: 32.
        activation_class (Type[nn.Module]): activation class to be used.
            default: nn.Tanh.
        use_td_params (bool, optional): if ``True``, the parameters can be found in `self.params` which is a
            :class:`~tensordict.nn.TensorDictParams` object (which inherits both from `TensorDict` and `nn.Module`).
            If ``False``, parameters are contained in `self._empty_net`. All things considered, these two approaches
            should be roughly identical but not interchangeable: for instance, a ``state_dict`` created with
            ``use_td_params=True`` cannot be used when ``use_td_params=False``.
        **kwargs: for :class:`torchrl.modules.models.MLP` can be passed to customize the MLPs.

    .. note:: to initialize the MARL module parameters with the `torch.nn.init`
        module, please refer to :meth:`~.get_stateful_net` and :meth:`~.from_stateful_net`
        methods.

    Examples:
        >>> from torchrl.modules import MultiAgentMLP
        >>> import torch
        >>> n_agents = 6
        >>> n_agent_inputs=3
        >>> n_agent_outputs=2
        >>> batch = 64
        >>> obs = torch.zeros(batch, n_agents, n_agent_inputs)
        >>> # instantiate a local network shared by all agents (e.g. a parameter-shared policy)
        >>> mlp = MultiAgentMLP(
        ...     n_agent_inputs=n_agent_inputs,
        ...     n_agent_outputs=n_agent_outputs,
        ...     n_agents=n_agents,
        ...     centralized=False,
        ...     share_params=True,
        ...     depth=2,
        ... )
        >>> print(mlp)
        MultiAgentMLP(
          (agent_networks): ModuleList(
            (0): MLP(
              (0): Linear(in_features=3, out_features=32, bias=True)
              (1): Tanh()
              (2): Linear(in_features=32, out_features=32, bias=True)
              (3): Tanh()
              (4): Linear(in_features=32, out_features=2, bias=True)
            )
          )
        )
        >>> assert mlp(obs).shape == (batch, n_agents, n_agent_outputs)
        Now let's instantiate a centralized network shared by all agents (e.g. a centalised value function)
        >>> mlp = MultiAgentMLP(
        ...     n_agent_inputs=n_agent_inputs,
        ...     n_agent_outputs=n_agent_outputs,
        ...     n_agents=n_agents,
        ...     centralized=True,
        ...     share_params=True,
        ...     depth=2,
        ... )
        >>> print(mlp)
        MultiAgentMLP(
          (agent_networks): ModuleList(
            (0): MLP(
              (0): Linear(in_features=18, out_features=32, bias=True)
              (1): Tanh()
              (2): Linear(in_features=32, out_features=32, bias=True)
              (3): Tanh()
              (4): Linear(in_features=32, out_features=2, bias=True)
            )
          )
        )
        We can see that the input to the first layer is n_agents * n_agent_inputs,
        this is because in the case the net acts as a centralized mlp (like a single huge agent)
        >>> assert mlp(obs).shape == (batch, n_agents, n_agent_outputs)
        Outputs will be identical for all agents.
        Now we can do both examples just shown but with an independent set of parameters for each agent
        Let's show the centralized=False case.
        >>> mlp = MultiAgentMLP(
        ...     n_agent_inputs=n_agent_inputs,
        ...     n_agent_outputs=n_agent_outputs,
        ...     n_agents=n_agents,
        ...     centralized=False,
        ...     share_params=False,
        ...     depth=2,
        ... )
        >>> print(mlp)
        MultiAgentMLP(
          (agent_networks): ModuleList(
            (0-5): 6 x MLP(
              (0): Linear(in_features=3, out_features=32, bias=True)
              (1): Tanh()
              (2): Linear(in_features=32, out_features=32, bias=True)
              (3): Tanh()
              (4): Linear(in_features=32, out_features=2, bias=True)
            )
          )
        )
        We can see that this is the same as in the first example, but now we have 6 MLPs, one per agent!
        >>> assert mlp(obs).shape == (batch, n_agents, n_agent_outputs)
    """

    def __init__(
        self,
        n_agent_inputs: int | None,
        n_agent_inputs_names: List[str] | None,
        n_agent_outputs: int,
        n_agents: int,
        *,
        centralized: bool | None = None,
        share_params: bool | None = None,
        device: Optional[DEVICE_TYPING] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        use_td_params: bool = True,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_inputs_names = n_agent_inputs_names
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.centralized = centralized
        self.num_cells = num_cells
        self.activation_class = activation_class
        self.depth = depth

        super().__init__(
            n_agents=n_agents,
            centralized=centralized,
            share_params=share_params,
            device=device,
            agent_dim=-2,
            use_td_params=use_td_params,
            **kwargs,
        )

    def _pre_forward_check(self, inputs):
        if inputs.shape[-2] != self.n_agents:
            raise ValueError(
                f"Multi-agent network expected input with shape[-2]={self.n_agents},"
                f" but got {inputs.shape}"
            )
        # If the model is centralized, agents have full observability
        if self.centralized:
            inputs = inputs.flatten(-2, -1)
        return inputs

    def _build_single_net(self, *, device, **kwargs):
        n_agent_inputs = self.n_agent_inputs
        if self.centralized and n_agent_inputs is not None:
            n_agent_inputs = self.n_agent_inputs * self.n_agents
        return MLP(
            in_features=n_agent_inputs,
            in_features_names=self.n_agent_inputs_names,
            out_features=self.n_agent_outputs,
            depth=self.depth,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
            device=device,
            **kwargs,
        )

class _GroupedByConfigLayers(nn.Module):
    
    """
    Grouped layers for a multi-layer perceptron.

    Args:

        in_feature_grouping (dict): dictionary containing the grouping of features and the layers for each group.
        device (torch.device): device to create the module on.

    """

    def __init__(self, in_feature_grouping, device):

        super(_GroupedByConfigLayers, self).__init__()

        self.in_feature_grouping = in_feature_grouping

        # Create layers for each group

        hidden_size = self.in_feature_grouping[list(self.in_feature_grouping.keys())[0]]["layers"][0].out_features
        n_groups = len(self.in_feature_grouping)

        for group in self.in_feature_grouping:

            setattr(self, group, nn.Sequential(*self.in_feature_grouping[group]["layers"]))

        self.output = nn.Linear(in_features=hidden_size * n_groups, out_features=hidden_size, device=device)

    def forward(self, x):

        outputs = []

        for group in self.in_feature_grouping:

            # Pick features by index
            features = x.index_select(-1, torch.tensor(self.in_feature_grouping[group]["indexes"], device=x.device))
            group_output = getattr(self, group)(features)
            outputs.append(group_output)

        return self.output(torch.cat(outputs, dim=-1))
