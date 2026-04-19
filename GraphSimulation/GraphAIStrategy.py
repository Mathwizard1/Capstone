from __future__ import annotations

from abc import ABC, abstractmethod

from GraphSimulation.GraphModel import TripartiteGraph
from GraphSimulation.utils import DEVICE

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

import os.path as path

from .utils import RND_GEN, DEVICE, DTYPE

from .GraphStrategy import MatchingStrategy

import torch.nn as nn

from torch import (
    Tensor,
    tensor,
    as_tensor,

    set_grad_enabled,

    cat,
    stack,

    full,
    zeros,
)

from torch import save as torch_save
from torch import load as torch_load

from torch.distributions import Categorical

from torchinfo import summary

import os

SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok= True)

# GraphAI Strategy Class

class BaseAIStrategy(nn.Module, MatchingStrategy, ABC):
    def __init__(self, name="BaseAIStrategy", deterministic_partner=True, is_recurrent= False) -> None:
        super().__init__()
        self.name = name
        self.deterministic_partner = deterministic_partner
        self.is_recurrent = is_recurrent

        self.actions = 0
        self.action_map: tuple[int,...] = ()

    @abstractmethod
    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> Tensor:
        pass

    @abstractmethod
    def process_graph(self, graph: TripartiteGraph):
        pass

    # Default inference for Nodes
    def select_inode_for_var(self, graph: TripartiteGraph, node: varNode):
        with set_grad_enabled(self.training):
            scores = self._get_inode_scores(graph, node)
                
        action = scores.softmax(dim=0).argmax()
        if action == self.actions: return None
        return graph.Inodes[self.action_map[action]]

    def select_inode_for_L(self, graph: TripartiteGraph, lnode: LNode) -> INode | None:
        return self.select_inode_for_var(graph, lnode)

    def select_inode_for_R(self, graph: TripartiteGraph, rnode: RNode) -> INode | None:
        return self.select_inode_for_var(graph, rnode)

    # For RL Algorithms
    def sample_action(self, scores: Tensor):
        probs = scores.softmax(dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def save(self, filepath=None, verbose= True):
        if filepath is None: filepath = f"{SAVE_DIR}/{self.name}.pth"

        torch_save(self.state_dict(), filepath)
        if(verbose): print(f"Model saved to {filepath}")

    def load(self, filepath=None):
        if filepath is None: filepath = f"{SAVE_DIR}/{self.name}.pth"

        if path.exists(filepath):
            self.load_state_dict(torch_load(filepath))
            print(f"Model loaded from {filepath}")
        else:
            print(f"Warning: No file found at {filepath}")

    def print_summary(self, input_size= None, depth= 5):
        print(summary(self, input_size=input_size, depth=depth))

class MLPStrategy(BaseAIStrategy):
    def __init__(self, 
                hidden_dim=64,
                embed_dim=32,
                device=DEVICE,
                name="MLPStrategy"):
        super().__init__(name, True, False)

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.device = device

        self.inode_embed_table: nn.Embedding = None # type: ignore
        self.base_mask: Tensor = None               # type: ignore

        # Encode edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, embed_dim),
        )

        # Encode inode features
        self.inode_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, embed_dim),
        )
        self.inode_gamma = nn.Parameter(zeros(1))
        self.tanh = nn.Tanh()

        # Global Projection layer
        self.global_projection = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        # Build query from aggregated edge info
        self.query_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, embed_dim)
        )

        # WAIT token
        self.wait_token = nn.Parameter(zeros(embed_dim))

    def process_graph(self, graph: TripartiteGraph):
        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)

        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]
        
        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        graph_state = graph.get_state(node)

        edge_feat = as_tensor(graph_state['edge'], device=self.device)           # (N, 4)
        inode_feat = as_tensor(graph_state['inode'], device=self.device)         # (N, 6)
        global_feat = as_tensor(graph_state['global'], device=self.device)     # (4,)

        edge_embed = self.edge_encoder(edge_feat)        # (N, E)
        inode_embed = self.inode_encoder(inode_feat)     # (N, E)
        global_proj = self.global_projection(global_feat)

        context = edge_embed.mean(dim=0)            # (E,)

        return context, inode_embed, global_proj

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        context, inode_embed, global_proj = self.update_state(graph, node)

        # Build query
        query_input = context + global_proj
        query = self.query_net(query_input)

        # Mask invalid actions
        mask = self.base_mask.clone()
        
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        # WAIT always valid
        mask[self.actions] = 0.0

        agg_inode_embed = self.inode_embed_table.weight + self.tanh(self.inode_gamma * inode_embed)
        action_embed = cat([agg_inode_embed, self.wait_token.unsqueeze(0)], dim= 0)

        scores = (action_embed * query.unsqueeze(0)).sum(dim=1)
        scores = scores + mask

        return scores

class ResidualMLPStrategy(MLPStrategy):
    def __init__(self,
                 hidden_dim=64,
                 embed_dim= 32,
                 block_dim= 16,
                 device=DEVICE,
                 name="ResidualMLPStrategy"):
        super().__init__(hidden_dim, embed_dim, device, name)

        self.block_dim = block_dim

        # Encode edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Edge encoder: input 6 features -> hidden_dim -> embed_dim using residual blocks
        self.edge_layer = nn.Linear(hidden_dim, embed_dim)
        self.edge_residual = self._residual_block(hidden_dim, embed_dim)
        self.edge_gamma = nn.Parameter(zeros(1))

        # Build query from aggregated edge info
        self.query_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.query_layer = nn.Linear(hidden_dim, embed_dim)
        self.query_residual = self._residual_block(hidden_dim, embed_dim)
        self.query_gamma = nn.Parameter(zeros(1))

    def _residual_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, self.block_dim),

            nn.LayerNorm(self.block_dim),
            nn.SiLU(),
            nn.Linear(self.block_dim, output_dim))

    def update_state(self, graph: TripartiteGraph, node: varNode):
        graph_state = graph.get_state(node)

        edge_feat = as_tensor(graph_state['edge'], device=self.device)           # (N, 4)
        inode_feat = as_tensor(graph_state['inode'], device=self.device)         # (N, 6)
        global_feat = as_tensor(graph_state['global'], device=self.device)       # (4,)

        edge_embed = self.edge_encoder(edge_feat)        # (N, E)
        inode_embed = self.inode_encoder(inode_feat)     # (N, E)
        global_proj = self.global_projection(global_feat)

        context = edge_embed.mean(dim=0)            # (E,)
        residual_context = self.edge_layer(context) + self.tanh(self.edge_gamma * self.edge_residual(context))

        return residual_context, inode_embed, global_proj

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        context, inode_embed, global_proj = self.update_state(graph, node)

        # Build query
        query_input = context + global_proj
        pre_query = self.query_net(query_input)
        query = self.query_layer(pre_query) + self.tanh(self.query_gamma * self.query_residual(pre_query))

        # Mask invalid actions
        mask = self.base_mask.clone()
        
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        # WAIT always valid
        mask[self.actions] = 0.0

        agg_inode_embed = self.inode_embed_table.weight + self.tanh(self.inode_gamma * inode_embed)
        action_embed = cat([agg_inode_embed, self.wait_token.unsqueeze(0)], dim= 0)

        scores = (action_embed * query.unsqueeze(0)).sum(dim=1)
        scores = scores + mask

        return scores

class CNNStrategy(BaseAIStrategy):
    def __init__(self,
                 embed_dim=32,
                 hidden_channels=16,
                 num_conv_layers=2,
                 device=DEVICE,
                 name="CNNStrategy"):
        super().__init__(name, True, False)

        self.embed_dim = embed_dim
        self.hidden_channels = hidden_channels
        self.num_conv_layers = num_conv_layers
        self.device = device

        self.inode_embed_table: nn.Embedding = None # type: ignore
        self.base_mask: Tensor = None               # type: ignore

        # Additional auxiliary features per INode 
        # (varNode.node_type, varNode.node_type, inode in varNode.candidate, 
        # left_count, right_count, balance, congestion)
        self.wait_aux = as_tensor([0.5, 0.0, 1.0, -1.0, -1.0, 0.0, -2.0])

        conv_layers = []
        in_ch = self.embed_dim + self.wait_aux.size(0)

        for i in range(num_conv_layers):
            out_ch = hidden_channels if i < num_conv_layers - 1 else 1
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))

            if i < num_conv_layers - 1:
                conv_layers.append(nn.BatchNorm1d(out_ch))
                conv_layers.append(nn.SiLU())

            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        self.inode_encoder = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        self.inode_gamma = nn.Parameter(zeros(1))
        self.tanh = nn.Tanh()

        # Learnable WAIT token embedding (will be concatenated with its auxiliary features)
        self.wait_embed = nn.Parameter(zeros(self.embed_dim))
        self.wait_token = cat([self.wait_embed.unsqueeze(0), self.wait_aux.unsqueeze(0)], dim=1)

    def process_graph(self, graph: TripartiteGraph):
        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)

        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]

        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        graph_state = graph.get_state(node)

        inode_feat = as_tensor(graph_state['inode'], device=self.device)    # (N,6)
        inode_embed = self.inode_encoder(inode_feat)
        agg_inode_embed = self.inode_embed_table.weight + self.tanh(self.inode_gamma * inode_embed)

        edge_feat = as_tensor(graph_state['edge'], device=self.device)     # (N,4)
        aux = cat([edge_feat[:, :3], inode_feat[:, 2:]], dim=1)
        action_features = cat([agg_inode_embed, aux], dim=1)

        x = cat([action_features, self.wait_token], dim=0)
        x = x.unsqueeze(0).permute(0, 2, 1)

        scores = self.conv_net(x).squeeze(0).squeeze(0)
        return scores

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        scores = self.update_state(graph, node)

        mask = self.base_mask.clone()

        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        mask[self.actions] = 0.0

        scores = scores + mask
        return scores

class TimeSeriesStrategy(BaseAIStrategy):
    def __init__(self,
            hidden_dim=64,
            embed_dim=32,
            state_dim=32,
            steps=50,
            device=DEVICE,
            name="TimeSeriesStrategy"):
        super().__init__(name, True, True)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.device = device

        self.inode_embed_table: nn.Embedding = None     # type: ignore
        self.action_embed: Tensor = None                # type: ignore
        self.base_mask: Tensor = None                   # type: ignore
        self.global_state: Tensor = None                # type: ignore

        self.BPTT_steps = steps
        self.step_counter = 0

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Global Projection layer
        self.global_projection = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        self.state_to_embed = nn.Linear(state_dim, embed_dim, bias=False)
        self.embed_to_state = nn.Linear(embed_dim, state_dim, bias=False)

        self.state_mask = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Sigmoid(),
            nn.Linear(embed_dim, state_dim),
        )

        self.query_net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.wait_token = nn.Parameter(zeros(embed_dim))

    def process_graph(self, graph: TripartiteGraph):
        self.global_state = zeros(self.state_dim, device=self.device)
        graph.embedding = self.global_state.detach()

        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)

        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]

        inode_embed = self.inode_embed_table.weight
        self.action_embed = cat([inode_embed, self.wait_token.unsqueeze(0)], dim=0)

        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        graph_state = graph.get_state(node)

        edge_feat = as_tensor(graph_state['edge'], device=self.device)
        global_feat = as_tensor(graph_state['global'], device=self.device)       # (4,)

        global_proj = self.global_projection(global_feat)

        edge_embed = self.edge_encoder(edge_feat)
        edge_t = edge_embed.mean(dim=0)

        h_t = edge_t + self.state_to_embed(self.global_state) + global_proj
        self.global_state = self.state_mask(h_t) * self.global_state + self.embed_to_state(h_t)

        self.step_counter += 1
        if self.step_counter > self.BPTT_steps:
            self.global_state = self.global_state.detach()
            self.step_counter = 0

        graph.embedding = self.global_state.detach()

        return edge_t, h_t

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        edge_t, h_t = self.update_state(graph, node)

        query = self.query_net(cat([edge_t, h_t]))

        mask = self.base_mask.clone()

        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        mask[self.actions] = 0.0

        scores = (self.action_embed * query.unsqueeze(0)).sum(dim=1)
        scores = scores + mask
        
        return scores

    def reset(self, graph: TripartiteGraph):
        self.global_state = zeros(self.state_dim, device=self.device)
        self.step_counter = 0
        graph.embedding = self.global_state.detach()

class TransformerStrategy(BaseAIStrategy):
    def __init__(self,
                 hidden_dim=64,
                 embed_dim=32,
                 num_heads=4,
                 num_layers=2,
                 device=DEVICE,
                 name="TransformerStrategy"):
        super().__init__(name, True, False)

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.device = device

        self.inode_embed_table: nn.Embedding = None          # type: ignore
        self.base_mask: Tensor = None                        # type: ignore

        self.input_proj = nn.Linear(10, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_projection = nn.Linear(4, embed_dim)

        self.query_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.wait_token = nn.Parameter(zeros(embed_dim))

    def process_graph(self, graph: TripartiteGraph):
        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)

        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]

        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        graph_state = graph.get_state(node)

        inode = as_tensor(graph_state['inode'], device=self.device)   # (N,6)
        edge = as_tensor(graph_state['edge'], device=self.device)     # (N,4)
        global_f = as_tensor(graph_state['global'], device=self.device)

        x = cat([inode, edge], dim=1)             # (N,10)
        x = self.input_proj(x).unsqueeze(0)       # (1,N,E)

        x = self.transformer(x).squeeze(0)        # (N,E)

        context = x.mean(dim=0)
        global_proj = self.global_projection(global_f)

        return context, x, global_proj

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        context, inode_embed, global_proj = self.update_state(graph, node)

        query = self.query_net(context + global_proj)

        action_embed = cat([inode_embed, self.wait_token.unsqueeze(0)], dim=0)

        mask = self.base_mask.clone()
        candidate_set = set(node.candidate_Inodes)

        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0

        mask[self.actions] = 0.0

        scores = (action_embed * query.unsqueeze(0)).sum(dim=1)
        return scores + mask