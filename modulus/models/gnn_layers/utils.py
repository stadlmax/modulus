# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Tuple, Union

import dgl.function as fn
import torch
from dgl import DGLGraph
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from modulus.models.gnn_layers.distributed_graph import DistributedGraph

try:
    from pylibcugraphops.pytorch import CSC
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_e2e,
    )

    USE_CUGRAPHOPS = True

except ImportError:
    raise ImportError("whut")
    CSC = None
    update_efeat_e2e = None
    agg_concat_e2n = None
    USE_CUGRAPHOPS = False


def checkpoint_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Applies the identity function for checkpointing.

    This function serves as an identity function for use with model layers
    when checkpointing is not enabled. It simply forwards the input arguments
    to the specified layer and returns its output.

    Parameters
    ----------
    layer : Callable
        The model layer or function to apply to the input arguments.
    *args
        Positional arguments to be passed to the layer.
    **kwargs
        Keyword arguments to be passed to the layer.

    Returns
    -------
    Any
        The output of the specified layer after processing the input arguments.
    """
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool) -> Callable:
    """Sets checkpoint function.

    This function returns the appropriate checkpoint function based on the
    provided `do_checkpointing` flag. If `do_checkpointing` is True, the
    function returns the checkpoint function from PyTorch's
    `torch.utils.checkpoint`. Otherwise, it returns an identity function
    that simply passes the inputs through the given layer.

    Parameters
    ----------
    do_checkpointing : bool
        Whether to use checkpointing for gradient computation. Checkpointing
        can reduce memory usage during backpropagation at the cost of
        increased computation time.

    Returns
    -------
    Callable
        The selected checkpoint function to use for gradient computation.
    """
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


def concat_message_function(edges: Tensor) -> Dict[str, Tensor]:
    """Concatenates source node, destination node, and edge features.

    Parameters
    ----------
    edges : Tensor
        Edges.

    Returns
    -------
    Dict[Tensor]
        Concatenated source node, destination node, and edge features.
    """
    # concats src node , dst node, and edge features
    cat_feat = torch.cat((edges.data["x"], edges.src["x"], edges.dst["x"]), dim=1)
    return {"cat_feat": cat_feat}


@torch.jit.ignore()
def concat_efeat_dgl(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[torch.Tensor, torch.Tensor]],
    graph: DGLGraph,
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor, Tensor]
        Node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    if isinstance(nfeat, Tuple):
        src_feat, dst_feat = nfeat
        with graph.local_scope():
            graph.srcdata["x"] = src_feat
            graph.dstdata["x"] = dst_feat
            graph.edata["x"] = efeat
            graph.apply_edges(concat_message_function)
            return graph.edata["cat_feat"]

    with graph.local_scope():
        graph.ndata["x"] = nfeat
        graph.edata["x"] = efeat
        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


def concat_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: Union[DGLGraph, CSC, DistributedGraph],
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features.
    graph : DGLGraph | CSC | DistributedGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """

    local_graph = graph
    src_feat = nfeat if isinstance(nfeat, Tensor) else nfeat[0]
    if isinstance(graph, DistributedGraph):
        local_graph = graph.local_graph
        src_feat = graph.get_src_node_features_in_local_graph(src_feat)
    
    if isinstance(nfeat, Tensor):
        nfeat = src_feat
        if USE_CUGRAPHOPS and isinstance(local_graph, CSC):
            res_efeat = update_efeat_e2e(
                efeat, nfeat, local_graph, mode="concat", use_source_emb=True, use_target_emb=True,
            )
        elif isinstance(local_graph, DGLGraph):
            src, dst = (item.long() for item in local_graph.edges())
            res_efeat = concat_efeat_dgl(efeat, nfeat, nfeat, src, dst)

        else:
            res_efeat = concat_efeat(efeat, nfeat, local_graph)

    else:
        dst_feat = nfeat[1]

        if USE_CUGRAPHOPS and isinstance(local_graph, CSC):
            res_efeat = update_efeat_e2e(
                efeat, (src_feat, dst_feat), local_graph, mode="concat", use_source_emb=True, use_target_emb=True,
            )

        elif isinstance(local_graph, DGLGraph):
            src, dst = (item.long() for item in local_graph.edges())
            res_efeat = concat_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
        
        else:
            res_efeat = concat_efeat(efeat, (src_feat, dst_feat), local_graph)

    return res_efeat


@torch.jit.script
def sum_efeat_dgl(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    src_idx : Tensor
        Source node indices.
    dst_idx : Tensor
        Destination node indices.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """

    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def sum_efeat(
    efeat: Tensor,
    nfeat: Union[Tensor, Tuple[Tensor]],
    graph: Union[DGLGraph, CSC, DistributedGraph],
):
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor | Tuple[Tensor]
        Node features (static setting) or tuple of node features of
        source and destination nodes (bipartite setting).
    graph : DGLGraph | CSC, DistributedGraph
        The underlying graph.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """

    local_graph = graph
    src_feat = nfeat if isinstance(nfeat, Tensor) else nfeat[0]
    if isinstance(local_graph, DistributedGraph):
        local_graph = graph.local_graph
        src_feat = graph.get_src_node_features_in_local_graph(src_feat)

    if isinstance(nfeat, Tensor):
        nfeat = src_feat
        if USE_CUGRAPHOPS and isinstance(local_graph, CSC):
            res_efeat = update_efeat_e2e(
                efeat, nfeat, local_graph, mode="sum", use_source_emb=True, use_target_emb=True,
            )
        elif isinstance(local_graph, DGLGraph):
            src, dst = (item.long() for item in local_graph.edges())
            res_efeat = sum_efeat_dgl(efeat, nfeat, nfeat, src, dst)

        else:
            res_efeat = sum_efeat(efeat, nfeat, local_graph)

    else:
        dst_feat = nfeat[1]

        if USE_CUGRAPHOPS and isinstance(local_graph, CSC):
            res_efeat = update_efeat_e2e(
                efeat, (src_feat, dst_feat), local_graph, mode="sum", use_source_emb=True, use_target_emb=True,
            )

        elif isinstance(local_graph, DGLGraph):
            src, dst = (item.long() for item in local_graph.edges())
            res_efeat = sum_efeat_dgl(efeat, src_feat, dst_feat, src, dst)
        
        else:
            res_efeat = sum_efeat(efeat, (src_feat, dst_feat), local_graph)

    return res_efeat


@torch.jit.ignore()
def agg_concat_dgl(
    efeat: Tensor, dst_nfeat: Tensor, graph: DGLGraph, aggregation: str
) -> Tensor:
    """Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = efeat

        # aggregate edge features
        if aggregation == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        # concat dst-node & edge features
        cat_feat = torch.cat((graph.dstdata["h_dest"], dst_nfeat), -1)
        return cat_feat


def aggregate_and_concat(
    efeat: Tensor,
    nfeat: Tensor,
    graph: Union[DGLGraph, CSC, DistributedGraph],
    aggregation: str,
):
    """
    Aggregates edge features and concatenates result with destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features (destination nodes).
    graph : DGLGraph | CSC | DistributedGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with destination node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    
    if USE_CUGRAPHOPS and isinstance(graph, CSC):
        cat_feat = agg_concat_e2n(nfeat, efeat, graph, aggregation)
    
    elif isinstance(graph, DGLGraph):
        cat_feat = agg_concat_dgl(efeat, nfeat, graph, aggregation)

    else:
        cat_feat = aggregate_and_concat(efeat, nfeat, graph.local_graph, aggregation)

    return cat_feat
