from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import exp, clip

from GraphSimulation.GraphModel import TripartiteGraph

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

from .GraphStrategy import MatchingStrategy
from .GraphAIStrategy import SAVE_DIR, BaseAIStrategy

from torch import (
    cuda,

    Tensor,
    tensor,

    long,
)

import os

import torch.optim as optim
import torch.nn as nn

from .utils import RND_GEN, DEVICE, DTYPE

from tqdm.auto import tqdm

class TripartiteGraphTrainer:
    def __init__(
        self,
        teacher: MatchingStrategy | BaseAIStrategy,
        student: BaseAIStrategy,
        n_inodes: int,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device=DEVICE,

        beta_decay= 0.5,
        beta_threshold= 0.45,
    ):
        self.teacher = teacher
        self.student = student

        # Two separate graphs
        self.teacher_graph = TripartiteGraph(teacher, n_inodes)
        self.student_graph = TripartiteGraph(student, n_inodes)

        self.teacher_inode_ids = tuple(self.teacher_graph.Inodes)
        self.student_inode_ids = tuple(self.student_graph.Inodes)
        self.student_id_to_idx = {id_: i for i, id_ in enumerate(self.student_inode_ids)}
        self.teacher_id_to_idx = {id_: i for i, id_ in enumerate(self.teacher_inode_ids)}

        self.optimizer = optimizer
        self.criterion = criterion

        self.loss_data = ()
        self.device = device

        self.beta = 1.0
        self.beta_decay = beta_decay
        self.beta_threshold = beta_threshold

    def _candidates_mapping(self, candidates, graph_inode_ids):
        inode_ids = [graph_inode_ids[candidate_id] for candidate_id in candidates]
        return tuple(inode_ids)

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

    def _dagger_policy(self, teacher_inode: INode|None, student_inode:INode|None, step):
        use_teacher = (RND_GEN.random() < self.beta)

        self.beta = (self.beta * exp(-self.beta_decay * step)).clip(self.beta_threshold)

        if use_teacher:
            exec_inode_teacher = teacher_inode
            if teacher_inode:
                t_idx = self.teacher_id_to_idx[teacher_inode.id]
                student_inode_id = self.student_inode_ids[t_idx]
                exec_inode_student = self.student_graph.Inodes[student_inode_id]
            else:
                exec_inode_student = None
        else:
            exec_inode_student = student_inode
            if student_inode:
                s_idx = self.student_id_to_idx[student_inode.id]
                teacher_inode_id = self.teacher_inode_ids[s_idx]
                exec_inode_teacher = self.teacher_graph.Inodes[teacher_inode_id]
            else:
                exec_inode_teacher = None
        return exec_inode_teacher, exec_inode_student

    def step_supervised(self, node_type, time, candidates) -> Tensor:
        # Create SAME node in both graphs
        t_node = self.teacher_graph.add_node(
            time,
            self._candidates_mapping(candidates, self.teacher_inode_ids),
            node_type
        )

        s_node = self.student_graph.add_node(
            time,
            self._candidates_mapping(candidates, self.student_inode_ids),
            node_type
        )

        # ---- Teacher (label) ----
        if node_type == "L":
            teacher_inode = self.teacher.select_inode_for_L(self.teacher_graph, t_node)
        else:
            teacher_inode = self.teacher.select_inode_for_R(self.teacher_graph, t_node)

        # ---- Student (prediction) ----
        scores = self.student.get_inode_scores(self.student_graph, s_node).unsqueeze(0)

        # ---- Map teacher → student index (for loss only) ----
        if teacher_inode:
            target_idx = self.teacher_id_to_idx[teacher_inode.id]
        else:
            target_idx = self.student.actions  # WAIT

        # Loss Computed
        target = tensor([target_idx], dtype=long, device=scores.device)
        loss = self.criterion(scores, target)

        # ---- Student (Action) ----
        student_action_idx = scores.softmax(dim=1).argmax(dim=1)

        if student_action_idx == self.student.actions:
            student_inode = None
        else:
            student_inode_id = self.student_inode_ids[student_action_idx]
            student_inode = self.student_graph.Inodes[student_inode_id]

        # ---- DAgger mixing (execution policy) ----
        exec_inode_teacher, exec_inode_student = self._dagger_policy(teacher_inode, student_inode, time)

        # ---- Apply SAME decision to both graphs ----
        self._apply_action(self.teacher_graph, self.teacher, t_node, exec_inode_teacher)
        self._apply_action(self.student_graph, self.student, s_node, exec_inode_student)

        return loss
    
    def train_supervised(self, node_order, epochs=10, accumulation_steps=10, 
                        save_model=False, save_dir=SAVE_DIR, verbose=True):
        os.makedirs(save_dir, exist_ok=True)
        best_loss = float('inf')
        loss_data = []
        n = len(node_order)

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()

            self.student.train()
            RND_GEN.shuffle(node_order)

            pbar = tqdm(enumerate(node_order), desc=f"Epoch {epoch + 1}", total=len(node_order), disable= not verbose)
            for time, (node_type, candidates) in pbar:
                loss = self.step_supervised(node_type, time, candidates)

                epoch_loss += loss.item()

                # Normalize loss for accumulation
                loss = loss / accumulation_steps
                loss.backward()

                # ---- UPDATE EVERY K STEPS ----
                if (time + 1) % accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), 1.25)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # ---- HANDLE REMAINDER ----
            nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

            avg_loss = epoch_loss / n
            loss_data.append(avg_loss)

            if(verbose): pbar.write(f"Epoch {epoch + 1}: avg_loss: {avg_loss:.4f}")
            pbar.close()

            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_model:
                    self.student.save(save_dir + f"/{self.student.name}_best.pth")

            self.teacher_graph.reset()
            self.student_graph.reset()

        self.loss_data = tuple(loss_data)
        self.student.save()

        print("Training done")
        return loss_data

    def plot_graph(self):
        plt.plot(self.loss_data, label=f"{self.student.name}")

        plt.title("Loss Data")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()
        plt.show()