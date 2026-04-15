from __future__ import annotations

from GraphSimulation.GraphModel import TripartiteGraph

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

from .GraphStrategy import MatchingStrategy
from .GraphAIStrategy import BaseAIStrategy

from torch import (
    cuda,

    Tensor,
    tensor,

    long,
)

import torch.optim as optim
import torch.nn as nn

from .utils import RND_GEN, DEVICE, DTYPE

class TripartiteGraphTrainer:
    def __init__(
        self,
        teacher: MatchingStrategy | BaseAIStrategy,
        student: BaseAIStrategy,
        n_inodes: int,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device=DEVICE
    ):
        self.teacher = teacher
        self.student = student

        # Two separate graphs
        self.teacher_graph = TripartiteGraph(teacher, n_inodes)
        self.student_graph = TripartiteGraph(student, n_inodes)

        self.teacher_inode_ids = tuple(self.teacher_graph.Inodes)
        self.student_inode_ids = tuple(self.student_graph.Inodes)

        self.optimizer = optimizer
        self.criterion = criterion

        self.device = device

    def _apply_action(self, graph: TripartiteGraph, strategy: MatchingStrategy, node: varNode, inode: INode|None):
        if node.node_type == 'L':
            if inode:
                partner = strategy.select_partner(graph, graph.right_memory[inode.id])
                if partner:
                    graph.match(node, inode, partner) # type: ignore
        elif node.node_type == 'R':
            if inode:
                partner = strategy.select_partner(graph, graph.left_memory[inode.id])
                if partner:
                    graph.match(partner, inode, node) # type: ignore

    def step_supervised(self, node_type, time, candidates) -> Tensor:
        # Create SAME node in both graphs
        t_node = self.teacher_graph.add_node(time, candidates, node_type)
        s_node = self.student_graph.add_node(time, candidates, node_type)

        # ---- Teacher ----
        if node_type == "L":
            teacher_inode = self.teacher.select_inode_for_L(self.teacher_graph, t_node)
        else:
            teacher_inode = self.teacher.select_inode_for_R(self.teacher_graph, t_node)

        student_inode = None
        if(teacher_inode):
            student_inode_id = self.student_inode_ids[self.teacher_inode_ids.index(teacher_inode.id)]
            student_inode = self.student_graph.Inodes[student_inode_id]

        # ---- Student ----
        scores = self.student.get_inode_scores(self.student_graph, s_node).unsqueeze(0)

        # ---- Target ----
        if teacher_inode and student_inode:
            target_idx = self.student_inode_ids.index(student_inode.id)
        else:
            target_idx = scores.shape[1] - 1  # WAIT action (last index)

        target = tensor([target_idx], dtype= long, device=scores.device)

        loss = self.criterion(scores, target)

        # ---- Apply actions ----
        self._apply_action(self.teacher_graph, self.teacher, t_node, teacher_inode)
        self._apply_action(self.student_graph, self.teacher, s_node, student_inode)

        return loss

    def train_supervised(self, node_order, epochs=10):
        for epoch in range(epochs):
            total_loss = tensor(0.0, device= self.device, requires_grad= True)

            RND_GEN.shuffle(node_order)
            for time, (node_type, candidates) in enumerate(node_order):
                loss = self.step_supervised(node_type, time, candidates)
                total_loss += loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1} | Loss: {total_loss.item():.4f}")

            self.teacher_graph.reset()
            self.student_graph.reset()