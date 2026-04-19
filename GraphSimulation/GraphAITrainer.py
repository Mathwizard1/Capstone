from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt

from numpy import (array, exp)

from GraphSimulation.GraphModel import TripartiteGraph
from GraphSimulation.utils import DEVICE

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
    as_tensor,

    stack,
    long,

    min,
)

import os

import torch.optim as optim
import torch.nn as nn

from .utils import RND_GEN, DEVICE, DTYPE, EPS

from tqdm.auto import tqdm

from abc import ABC, abstractmethod

# Value Nets
# TODO: For PPO / A2C
class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

# RL Policies
class BaseRLPolicy(ABC):
    def __init__(self) -> None:
        super().__init__()

        self.log_probs = []
        self.rewards = []
        self.entropies = []

    @abstractmethod
    def compute_loss(self) -> Tensor: ...

    def reset_episode(self):
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def store_step(self, log_prob, reward, entropy):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)
    
    def finish_episode(self, graph: TripartiteGraph):
        if len(self.rewards) > 0:
            self.rewards[-1] += graph.matches

    def compute_reward(self, graph: TripartiteGraph, node: varNode, inode: INode|None) -> float:
        reward = 0.0

        # ---- INVALID ACTION ----
        if inode is None:
            # WAIT penalty if valid actions exist
            if any(graph.Inodes[i].available for i in node.candidate_Inodes):
                reward -= 0.75
            else:
                reward += 0.5  # correct WAIT
            return reward

        # ---- MATCH POTENTIAL ----
        if node.node_type == 'L':
            partners = graph.right_memory[inode.id]
        else:
            partners = graph.left_memory[inode.id]

        if partners:
            reward += 2.5  # successful match opportunity
        else:
            reward += 1.0  # valid but not useful yet

        # ---- LOAD BALANCING ----
        degree = len(graph.left_memory[inode.id]) + len(graph.right_memory[inode.id])
        reward -= 0.05 * degree  # discourage overuse

        return reward
# ---------------- VanillaPolicy ---------------- #

class VanillaPolicyGradient(BaseRLPolicy):
    def __init__(self, gamma=0.99, entropy_beta=0.01, device=DEVICE):
        super().__init__()

        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.device = device

    def _compute_returns(self):
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return as_tensor(returns, device=self.device, dtype=DTYPE)

    def compute_loss(self):
        returns = self._compute_returns()
        returns = (returns - returns.mean()) / (returns.std() + EPS)

        log_probs = stack(self.log_probs)
        entropies = stack(self.entropies)

        policy_loss = -(log_probs * returns).sum()
        entropy_loss = -self.entropy_beta * entropies.sum()

        return policy_loss + entropy_loss
# ---------------- A2C ---------------- #

class A2CPolicy(VanillaPolicyGradient):
    def __init__(self, value_net: ValueNet, gamma=0.99, entropy_beta=0.01, value_beta=0.5, device=DEVICE):
        super().__init__(gamma, entropy_beta, device)

        self.value_beta = value_beta
        self.value_net = value_net
        self.values = []

    def reset_episode(self):
        self.values.clear()
        super().reset_episode()

    def compute_loss(self):
        returns = self._compute_returns()

        values = stack(self.values).squeeze()
        log_probs = stack(self.log_probs)
        entropies = stack(self.entropies)

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).sum()
        value_loss = ((returns - values) ** 2).sum()
        entropy_loss = -self.entropy_beta * entropies.sum()

        return policy_loss + self.value_beta * value_loss + entropy_loss

    def compute_reward(self, graph: TripartiteGraph, node: varNode, inode: INode):
        state = graph.get_state(node)
        value = self.value_net(state)
        self.values.append(value)

        return super().compute_reward(graph, node, inode)
# ---------------- PPO ---------------- #

class PPOPolicy(A2CPolicy):
    def __init__(self, value_net: ValueNet, gamma=0.99, clip_eps=0.2, entropy_beta=0.01, value_beta=0.5, device=DEVICE):
        super().__init__(value_net, gamma, entropy_beta, value_beta, device)

        self.clip_eps = clip_eps
        self.old_log_probs = []

    def reset_episode(self):
        self.old_log_probs.clear()
        super().reset_episode()

    def store_step(self, log_prob, reward, entropy):
        self.old_log_probs.append(log_prob.detach())
        super().store_step(log_prob, reward, entropy)

    def compute_loss(self):
        returns = self._compute_returns()

        values = stack(self.values).squeeze()
        log_probs = stack(self.log_probs)
        old_log_probs = stack(self.old_log_probs)
        entropies = stack(self.entropies)

        advantages = returns - values.detach()

        ratios = (log_probs - old_log_probs).exp()

        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = min(surr1, surr2).sum()
        value_loss = ((returns - values) ** 2).sum()
        entropy_loss = -self.entropy_beta * entropies.sum()

        return policy_loss + self.value_beta * value_loss + entropy_loss

# GraphTrainer Class

class TripartiteGraphTrainer:
    def __init__(
        self,
        n_Inodes: int,
        criterion: nn.Module,
        device=DEVICE,

        beta_decay= 5e-2,
        beta_threshold= 0.6,
        beta_decay_func: Literal['linear', 'exponential']= 'linear',
    ):
        self.teacher: MatchingStrategy | BaseAIStrategy = None  # type: ignore
        self.student: BaseAIStrategy = None                     # type: ignore

        # Two separate graphs
        self.teacher_graph: TripartiteGraph = None              # type: ignore
        self.student_graph: TripartiteGraph = None              # type: ignore

        self.teacher_inode_ids = tuple()
        self.student_inode_ids = tuple()
        self.student_id_to_idx = {}
        self.teacher_id_to_idx = {}

        self.n_inodes = n_Inodes
        self.optimizer: optim.Optimizer = None                  # type: ignore
        self.criterion = criterion

        self.loss_data = {}
        self.device = device

        self.beta = array(1.0)
        self.beta_decay = beta_decay
        self.beta_threshold = beta_threshold
        self.beta_decay_func = beta_decay_func

        # For RL basedTraining
        self.rl_policy: BaseRLPolicy = None                     # type: ignore

    def set_teacher(self, teacher:MatchingStrategy | BaseAIStrategy):
        self.teacher = teacher

        # graphs
        self.teacher_graph = TripartiteGraph(teacher, self.n_inodes)
        self.teacher_inode_ids = tuple(self.teacher_graph.Inodes)
        self.teacher_id_to_idx = {id_: i for i, id_ in enumerate(self.teacher_inode_ids)}

    def set_student(self, student:BaseAIStrategy, optimizer: optim.Optimizer):
        self.student = student
        self.optimizer = optimizer

        # graphs
        self.student_graph = TripartiteGraph(student, self.n_inodes)
        self.student_inode_ids = tuple(self.student_graph.Inodes)
        self.student_id_to_idx = {id_: i for i, id_ in enumerate(self.student_inode_ids)}

    def set_rl_policy(self, rl_policy: BaseRLPolicy):
        self.rl_policy = rl_policy

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
        if(self.beta_decay_func == 'linear'):
            beta = (self.beta - self.beta_decay * step).clip(self.beta_threshold)
        else:
            beta = (self.beta * exp(-self.beta_decay * step)).clip(self.beta_threshold)
        
        use_teacher = (RND_GEN.random() < beta)
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

    def step_supervised(self, node_type, time, candidates, epoch) -> Tensor:
        # ---- Create SAME node in both graphs ----
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

        # ---- Teacher scores (numpy or tensor → tensor) ----
        teacher_scores = self.teacher._get_inode_scores(self.teacher_graph, t_node)

        if not isinstance(teacher_scores, Tensor):
            teacher_scores[teacher_scores == 0.0] = -float('inf')
            teacher_scores = tensor(teacher_scores, device=self.device)
        else:
            teacher_scores = teacher_scores.to(self.device)

        teacher_scores = teacher_scores.unsqueeze(0).detach()

        # ---- Student scores ----
        student_scores = self.student._get_inode_scores(self.student_graph, s_node)
        student_scores = student_scores.unsqueeze(0)

        # ---- Loss (distribution-based) ----
        loss = self.criterion(student_scores, teacher_scores)

        # ---- Deterministic actions (argmax) ----
        teacher_idx = teacher_scores.argmax(dim=1)
        student_idx = student_scores.detach().argmax(dim=1)

        # ---- Map to inodes ----
        teacher_inode = (None if teacher_idx == self.student.actions
            else self.teacher_graph.Inodes[self.teacher_inode_ids[teacher_idx]]
        )

        student_inode = (None if student_idx == self.student.actions
            else self.student_graph.Inodes[self.student_inode_ids[student_idx]]
        )

        # ---- DAgger policy ----
        exec_inode_teacher, exec_inode_student = self._dagger_policy(teacher_inode, student_inode, epoch)

        # ---- Apply decisions ----
        self._apply_action(self.teacher_graph, self.teacher, t_node, exec_inode_teacher)
        self._apply_action(self.student_graph, self.student, s_node, exec_inode_student)

        return loss
    
    def train_supervised(self, node_order, epochs=10, 
                        save_model=False, save_dir=SAVE_DIR, verbose=True):
        os.makedirs(save_dir, exist_ok=True)

        loss_data = []
        n = len(node_order)

        self.teacher_graph.reset()
        self.student_graph.reset()

        for epoch in range(epochs):
            epoch_loss = 0.0
            episode_loss = []

            self.student.train()
            RND_GEN.shuffle(node_order)

            self.optimizer.zero_grad()

            pbar = tqdm(enumerate(node_order), desc=f"Epoch {epoch + 1}", total=len(node_order), disable= not verbose)
            for time, (node_type, candidates) in pbar:
                loss = self.step_supervised(node_type, time, candidates, epoch)

                epoch_loss += loss.item()
                episode_loss.append(loss)

            total_loss = stack(episode_loss).mean()
            total_loss.backward()

            # ---- Backpropagation ----
            nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            avg_loss = epoch_loss / n
            loss_data.append(avg_loss)

            if(verbose): pbar.write(f"Epoch {epoch + 1}: avg_loss: {avg_loss:.4f}")
            pbar.close()

            if save_model and (epoch % 20 == 0):
                self.student.save(save_dir + f"/{self.student.name}_best.pth", verbose= verbose)

            self.teacher_graph.reset()
            self.student_graph.reset()

        self.loss_data[f"{self.teacher.name}->{self.student.name}"] = tuple(loss_data)
        self.student.save()

        print("Training done")

    def step_rl(self, node_type, time, candidates):
        # ---- Create node ----
        s_node = self.student_graph.add_node(
            time,
            self._candidates_mapping(candidates, self.student_inode_ids),
            node_type
        )

        # ---- Get scores ----
        scores = self.student._get_inode_scores(self.student_graph, s_node)

        # ---- Sample action ----
        action, log_prob, entropy = self.student.sample_action(scores)

        # ---- Map action to inode ----
        if action == self.student.actions:
            chosen_inode = None
        else:
            inode_id = self.student_inode_ids[action]
            chosen_inode = self.student_graph.Inodes[inode_id]

        # ---- Apply action ----
        self._apply_action(self.student_graph, self.student, s_node, chosen_inode)

        # ---- Compute reward ----
        reward = self.rl_policy.compute_reward(self.student_graph, s_node, chosen_inode)

        # ---- Store trajectory ----
        self.rl_policy.store_step(log_prob, reward, entropy)

        return reward

    def train_rl(self, node_order, epochs=10,
                save_model=False, save_dir=SAVE_DIR, verbose=True):
        os.makedirs(save_dir, exist_ok=True)

        reward_data = []
        n = len(node_order)

        self.student_graph.reset()

        for epoch in range(epochs):
            total_reward = 0.0

            self.student.train()
            RND_GEN.shuffle(node_order)

            self.optimizer.zero_grad()
            self.rl_policy.reset_episode()

            pbar = tqdm(enumerate(node_order), desc=f"Epoch {epoch + 1}", total=n, disable=not verbose)
            for time, (node_type, candidates) in pbar:
                reward = self.step_rl(node_type, time, candidates)
                total_reward += reward

            # ---- Compute Reward & RL loss ----
            self.rl_policy.finish_episode(self.student_graph)
            loss = self.rl_policy.compute_loss()

            # ---- Backprop ----
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            avg_reward = total_reward / n
            reward_data.append(avg_reward)

            if verbose: pbar.write(f"Epoch {epoch + 1}: avg_reward: {avg_reward:.4f}")
            pbar.close()

            if save_model and (epoch % 20 == 0):
                self.student.save(save_dir + f"/{self.student.name}_rl.pth", verbose=verbose)

            self.student_graph.reset()

        self.loss_data[f"RL->{self.student.name}"] = tuple(reward_data)
        self.student.save()

        print("RL Training done")

    def plot_graph(self):
        for key in self.loss_data:
            plt.plot(self.loss_data[key], label=key)

        plt.title("Loss Data")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()
        plt.show()