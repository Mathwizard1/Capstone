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

    cat,
    stack,

    zeros,
)

from .utils import DEVICE, DTYPE

from torch import save as torch_save
from torch import load as torch_load

from torch.distributions import Categorical

# GraphAI Strategy Class

class BaseAIStrategy(nn.Module, MatchingStrategy, ABC):
    def __init__(self, name="BaseAIStrategy") -> None:
        super().__init__()
        self.name = name

        self.actions = 0
        self.action_map: tuple[int,...] = ()

    @abstractmethod
    def process_graph(self, graph: TripartiteGraph):
        pass

    @abstractmethod
    def get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> Tensor:
        pass

    # Default inference for Nodes
    def select_inode_for_var(self, graph: TripartiteGraph, node: varNode):
        scores = self.get_inode_scores(graph, node)
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
        return action.item(), dist.log_prob(action), dist.entropy()

    def save(self, filepath=None):
        if filepath is None: filepath = f"{self.name}.pth"

        torch_save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath=None):
        if filepath is None: filepath = f"{self.name}.pth"

        if path.exists(filepath):
            self.load_state_dict(torch_load(filepath))
            print(f"Model loaded from {filepath}")
        else:
            print(f"Warning: No file found at {filepath}")

class TimeSeriesStrategy(BaseAIStrategy):
    def __init__(self, 
            state_dim= 32, hidden_dim= 16, embed_dim= 16,
            device= DEVICE,
            name= "TimeSeriesStrategy") -> None:
        super().__init__(name)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.device = device

        self.inode_embed_table: nn.Embedding | None = None
        self.global_state = nn.Parameter(zeros(self.state_dim))

        self.edge_encoder = nn.Sequential(
            nn.Linear(3 + 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.embed_dim),
        )

        self.time_series_model = nn.GRUCell(self.embed_dim, self.embed_dim)

        # State and Embedding conversions
        self.state_to_embed = nn.Linear(self.state_dim, self.embed_dim, bias= False)
        self.embed_to_state = nn.Linear(self.embed_dim, self.state_dim, bias= False)
        #self.state_to_embed.weight = self.embed_to_state.weight

        self.state_mask = nn.Sequential(
            nn.Linear(self.embed_dim, self.state_dim),
            nn.Sigmoid(),
        )

        self.wait_token = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        self.query_token = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),

            nn.Linear(self.hidden_dim, self.embed_dim)
        )

    def process_graph(self, graph: TripartiteGraph):
        graph.embedding = self.global_state.clone()

        self.actions = len(graph.Inodes)
        self.action_map = tuple(graph.Inodes)

        # Register embedding module
        self.inode_embed_table = nn.Embedding(self.actions, self.embed_dim)
        self.inode_embed_table.to(self.device)

        for idx, inode in enumerate(graph.Inodes.values()):
            inode.rank = RND_GEN.random()
            inode.embedding = self.inode_embed_table.weight[idx]

    def embed_edge(self, node: varNode, inode: INode, left_count, right_count):
        inode_feat = tensor([
            float(node.online_time),
            float(len(node.candidate_Inodes)),
            0.0 if node.node_type == 'L' else 1.0,
            float(inode.available),
            float(inode.rank),
            float(left_count),
            float(right_count),
        ], device= self.device)

        return self.edge_encoder(inode_feat)

    def update_state(self, graph: TripartiteGraph, node: varNode) -> tuple[Tensor, Tensor]:
        edge_feat = []
        
        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]

            left_count, right_count = -1, -1
            if(inode.available):
                left_count = len(graph.left_memory[inode_id])
                right_count = len(graph.right_memory[inode_id])

            # Edge embedding
            edge_embeddings = self.embed_edge(node, inode, left_count, right_count)
            edge_feat.append(edge_embeddings)

        # Edge mean
        edge_t = stack(edge_feat).mean(dim= 0)
        
        h_t = self.state_to_embed(graph.embedding)
        h_t = self.time_series_model(edge_t, h_t)

        graph.embedding = (self.state_mask(h_t) * graph.embedding + self.embed_to_state(h_t)).detach()
        return edge_t, h_t

    def get_inode_scores(self, graph: TripartiteGraph, node: varNode):
        edge_t, h_t = self.update_state(graph, node)

        # Combine local + global signal
        query = self.query_token(cat([edge_t, h_t]))
        wait_embed = self.wait_token(h_t)

        inode_embeds = []
        mask = []

        for inode_id in self.action_map:
            inode = graph.Inodes[inode_id]

            inode_embeds.append(inode.embedding)

            # valid only if available AND candidate
            if inode.available and inode_id in node.candidate_Inodes:
                mask.append(0.0)
            else:
                mask.append(-float('inf'))

        inode_embeds.append(wait_embed)
        mask.append(0.0)

        inode_embeds = stack(inode_embeds)

        # Compute scores (dot product)
        scores = (inode_embeds * query.unsqueeze(0)).sum(dim=1) + tensor(mask, device= query.device)
        return scores