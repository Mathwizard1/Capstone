from __future__ import annotations

from abc import ABC, abstractmethod

from GraphSimulation.GraphModel import TripartiteGraph

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

import os.path as path

from .utils import RND_GEN

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

from .utils import DEVICE, DTYPE

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
        self.action_embed: Tensor = None            # type: ignore
        self.base_mask: Tensor = None               # type: ignore

        # Encode edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(3 + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, embed_dim),
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

        inode_embeds = self.inode_embed_table.weight
        self.action_embed = cat([inode_embeds, self.wait_token.unsqueeze(0)], dim=0)
        
        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        edge_feature = []

        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            left_cnt = len(graph.left_memory[inode_id]) if inode.available else -1
            right_cnt = len(graph.right_memory[inode_id]) if inode.available else -1
            edge_feature.append([
                0.0 if node.node_type == 'L' else 1.0,
                float(node.online_time),
                float(len(node.candidate_Inodes)),
                float(left_cnt),
                float(right_cnt),
                float(inode.state),
            ])
        edge_tensors = as_tensor(edge_feature, device=self.device)
        node_embeddings = self.edge_encoder(edge_tensors).mean(dim= 0)
        return node_embeddings

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        node_embed = self.update_state(graph, node)

        # Build query
        query = self.query_net(node_embed)

        # Mask invalid actions
        mask = self.base_mask.clone()
        mask.fill_(-float('inf'))
        
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        # WAIT always valid
        mask[self.actions] = 0.0

        scores = (self.action_embed * query.unsqueeze(0)).sum(dim=1)
        scores = scores + mask

        return scores

class ResidualMLPStrategy(BaseAIStrategy):
    def __init__(self,
                 hidden_dim=64,
                 embed_dim= 32,
                 block_dim= 16,
                 device=DEVICE,
                 name="ResidualMLPStrategy"):
        super().__init__(name, True, False)

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.block_dim = block_dim

        self.device = device

        self.inode_embed_table: nn.Embedding = None     # type: ignore
        self.action_embed: Tensor = None                # type: ignore
        self.base_mask: Tensor = None                   # type: ignore

        # Encode edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(3 + 3, hidden_dim),
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

        # WAIT token (learnable)
        self.wait_token = nn.Parameter(zeros(embed_dim))

    def _residual_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, self.block_dim),

            nn.LayerNorm(self.block_dim),
            nn.SiLU(),
            nn.Linear(self.block_dim, output_dim))

    def process_graph(self, graph: TripartiteGraph):
        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)
        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]

        inode_embeds = self.inode_embed_table.weight
        self.action_embed = cat([inode_embeds, self.wait_token.unsqueeze(0)], dim=0)

        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        edge_feature = []
        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            left_cnt = len(graph.left_memory[inode_id]) if inode.available else -1
            right_cnt = len(graph.right_memory[inode_id]) if inode.available else -1
            edge_feature.append([
                0.0 if node.node_type == 'L' else 1.0,
                float(node.online_time),
                float(len(node.candidate_Inodes)),
                float(left_cnt),
                float(right_cnt),
                float(inode.state),
            ])
        edge_tensors = as_tensor(edge_feature, device=self.device)
        node_embedding = self.edge_encoder(edge_tensors).mean(dim=0)

        final_embedding = self.edge_layer(node_embedding) + self.edge_gamma * self.edge_residual(node_embedding)
        return final_embedding

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        node_embed = self.update_state(graph, node)
        node_query = self.query_net(node_embed)  
        
        # residual MLP transforms the query
        query = self.query_layer(node_query) + self.query_gamma * self.query_residual(node_query)

        mask = self.base_mask.clone()
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        mask[self.actions] = 0.0  # WAIT always valid

        scores = (self.action_embed * query.unsqueeze(0)).sum(dim=1)
        scores = scores + mask
        return scores

class CNNStrategy(BaseAIStrategy):
    def __init__(self,
                 embed_dim= 32,
                 hidden_channels= 16,
                 num_conv_layers= 2,
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
        # left_count, right_count, available)
        self.wait_aux = as_tensor([0.5, 0.0, 1.0, -1.0, -1.0, 1.0])  # WAIT is always available

        conv_layers = []
        in_ch = self.embed_dim + self.wait_aux.size(0)
        for i in range(num_conv_layers):
            out_ch = hidden_channels if i < num_conv_layers - 1 else 1
            conv_layers.append(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=1, padding="same"))
            
            if i < num_conv_layers - 1:
                conv_layers.append(nn.BatchNorm1d(out_ch))
                conv_layers.append(nn.SiLU())
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        # Learnable WAIT token embedding (will be concatenated with its auxiliary features)
        self.wait_embed = nn.Parameter(zeros(embed_dim))

    def process_graph(self, graph: TripartiteGraph):
        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)
        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]

        # Mask for invalid actions (WAIT is always allowed)
        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode):
        action_features = [] # (num_actions+1, embed_dim + aux_dim)

        # Features for INodes only
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            left_cnt = len(graph.left_memory[inode_id]) if inode.available else -1.0
            right_cnt = len(graph.right_memory[inode_id]) if inode.available else -1.0
            
            aux = as_tensor([float(0.0 if node.node_type == 'L' else 1.0), 
                            float(node.online_time),
                            float(1.0 if inode_id in candidate_set else 0.0),
                            float(left_cnt), 
                            float(right_cnt), 
                            float(inode.state)], 
                            device=self.device)
            action_features.append(cat([inode.embedding, aux]))

        # Wait action
        action_features.append(cat([self.wait_embed, self.wait_aux.to(self.device)]))

        # Stack: (seq_len, in_channels) -> (1, in_channels, seq_len) for Conv1d
        x = stack(action_features, dim=0).unsqueeze(0).permute(0, 2, 1)  # (1, C, L)

        # Apply 1D convolutions -> (1, 1, L)
        scores = self.conv_net(x).squeeze(0).squeeze(0)  # (L,)
        return scores

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        scores = self.update_state(graph, node)

        mask = self.base_mask.clone()
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        mask[self.actions] = 0.0  # WAIT always valid

        scores = scores + mask
        return scores

class TimeSeriesStrategy(BaseAIStrategy):
    def __init__(self,
            hidden_dim= 64,
            embed_dim= 32,
            state_dim= 32,
            steps= 50,
            device= DEVICE,
            name= "TimeSeriesStrategy") -> None:
        super().__init__(name, True, True)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.device = device

        self.inode_embed_table: nn.Embedding = None # type: ignore
        self.base_mask: Tensor = None               # type: ignore
        self.global_state: Tensor = None            # type: ignore

        # BPTT
        self.BPTT_steps = steps
        self.step_counter = 0

        self.edge_encoder = nn.Sequential(
            nn.Linear(3 + 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.embed_dim),
        )

        # State and Embedding conversions
        self.state_to_embed = nn.Linear(self.state_dim, self.embed_dim, bias= False)
        self.embed_to_state = nn.Linear(self.embed_dim, self.state_dim, bias= False)

        self.state_mask = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Sigmoid(),
            nn.Linear(self.embed_dim, self.state_dim),
        )

        self.query_net = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),

            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # WAIT token (learnable)
        self.wait_token = nn.Parameter(zeros(embed_dim))

    def process_graph(self, graph: TripartiteGraph):
        self.global_state = zeros(self.state_dim, device=self.device)
        graph.embedding = self.global_state.detach()

        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim, device=self.device)

        for idx, inode in enumerate(graph.Inodes.values()):
            inode.embedding = self.inode_embed_table.weight[idx]

        inode_embeds = self.inode_embed_table.weight
        self.action_embed = cat([inode_embeds, self.wait_token.unsqueeze(0)], dim=0)

        self.base_mask = full((self.actions + 1,), -float('inf'), device=self.device, requires_grad=False)

    def update_state(self, graph: TripartiteGraph, node: varNode) -> tuple[Tensor, Tensor]:
        edge_feature = []

        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            left_cnt = len(graph.left_memory[inode_id]) if inode.available else -1
            right_cnt = len(graph.right_memory[inode_id]) if inode.available else -1
            edge_feature.append([
                0.0 if node.node_type == 'L' else 1.0,
                float(node.online_time),
                float(len(node.candidate_Inodes)),
                float(left_cnt),
                float(right_cnt),
                float(inode.state),
            ])
        feat_tensor = as_tensor(edge_feature, device=self.device)   # ONE tensor
        edge_embeddings = self.edge_encoder(feat_tensor)            # BATCHED

        edge_t = edge_embeddings.mean(dim=0)
        
        # Memory for context
        h_t = edge_t + self.state_to_embed(self.global_state)
        self.global_state = self.state_mask(h_t) * self.global_state + self.embed_to_state(h_t)

        # --- BPTT truncation ---
        self.step_counter += 1
        if self.step_counter > self.BPTT_steps:
            self.global_state = self.global_state.detach()
            self.step_counter = 0

        graph.embedding = self.global_state.detach()
        return edge_t, h_t
    
    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        edge_t, h_t = self.update_state(graph, node)

        # Build query
        query = self.query_net(cat([edge_t, h_t]))

        # Mask invalid actions
        mask = self.base_mask.clone()
        mask.fill_(-float('inf'))
        
        candidate_set = set(node.candidate_Inodes)
        for idx, inode_id in enumerate(self.action_map):
            inode = graph.Inodes[inode_id]
            if inode.available and inode_id in candidate_set:
                mask[idx] = 0.0
        # WAIT always valid
        mask[self.actions] = 0.0

        scores = (self.action_embed * query.unsqueeze(0)).sum(dim=1)
        scores = scores + mask

        return scores

    def reset(self, graph: TripartiteGraph):
        self.global_state = zeros(self.state_dim, device=self.device)
        self.step_counter = 0

        graph.embedding = self.global_state.detach()