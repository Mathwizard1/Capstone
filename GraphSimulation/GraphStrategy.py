from __future__ import annotations

from abc import ABC, abstractmethod

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

from .utils import RND_GEN

from numpy import (
    ndarray,
    array,

    exp,
)

# Matching Strategy Class

class MatchingStrategy(ABC):
    def __init__(self, name = "MatchingStrategy", deterministic_partner= True) -> None:
        self.name = name
        self.deterministic_partner = deterministic_partner

    @abstractmethod
    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray: ...

    def process_graph(self, graph: TripartiteGraph): 
        pass

    @abstractmethod
    def select_inode_for_L(self, graph: TripartiteGraph, lnode: LNode) -> INode | None: ...

    @abstractmethod
    def select_inode_for_R(self, graph: TripartiteGraph, rnode: RNode) -> INode | None: ...

    def select_partner(self, graph: TripartiteGraph, nodes: set[varNode]):
        if not nodes:
            return None
        return next(iter(sorted(nodes)))

    def reset(self, graph:TripartiteGraph):
        pass

class RandomStrategy(MatchingStrategy):
    def __init__(self, name="RandomStrategy", deterministic_partner=False) -> None:
        super().__init__(name, deterministic_partner)

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode_scores = []

        has_valid = False
        candidate_set = set(node.candidate_Inodes)
        for inode_id in graph.Inodes:
            inode = graph.Inodes[inode_id]
            valid = inode.available and (inode_id in candidate_set)

            if valid:
                inode_scores.append(1.0)
                has_valid = True
            else:
                inode_scores.append(0.0)

        # WAIT action
        inode_scores.append(0.0 if has_valid else 1.0)

        scores = array(inode_scores, dtype=float)
        return scores

    def _get_random_available_inode(self, graph: TripartiteGraph, node: varNode) -> INode | None:
        available_candidates = []
        
        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                available_candidates.append(inode)

        # if no candidates are free
        if not available_candidates: return None
        idx = RND_GEN.integers(len(available_candidates))
        return available_candidates[idx]

    def select_inode_for_L(self, graph, lnode):
        return self._get_random_available_inode(graph, lnode)

    def select_inode_for_R(self, graph, rnode):
        return self._get_random_available_inode(graph, rnode)

    def select_partner(self, graph, nodes: set[varNode]):
        if(self.deterministic_partner):
            return super().select_partner(graph, nodes)

        if not nodes:
            return None

        node_tuple = tuple(nodes)
        idx = RND_GEN.integers(len(node_tuple))
        return node_tuple[idx]

class GreedyStrategy(MatchingStrategy):
    def __init__(self, name="GreedyStrategy",) -> None:
        super().__init__(name, True)

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode= None

        for inode_id in node.candidate_Inodes:
            inode= graph.Inodes[inode_id]
            if(inode.available): 
                break

        inode_scores = list([0.0] * len(graph.Inodes))

        # WAIT action
        if(inode):
            inode_idx = tuple(graph.Inodes.keys()).index(inode.id)

            inode_scores[inode_idx] = 1.0
            inode_scores.append(0.0)
        else:
            inode_scores.append(1.0)

        scores = array(inode_scores, dtype=float)
        return scores

    def select_inode_for_L(self, graph, lnode):
        # Optimal with R connected
        for inode_id in lnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                return inode
        return None

    def select_inode_for_R(self, graph, rnode):
        # Optimal with L connected
        for inode_id in rnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                return inode
        return None

class RankStrategy(MatchingStrategy):
    def __init__(self, name= "RankStrategy", deterministic_partner= False) -> None:
        super().__init__(name, deterministic_partner)

    def _get_inode_scores(self, graph: TripartiteGraph, node: varNode) -> ndarray:
        inode_scores = []

        candidate_ids = [
            inode_id for inode_id in node.candidate_Inodes
            if graph.Inodes[inode_id].available
        ]

        # Case 4: no available neighbors → WAIT
        if not candidate_ids:
            scores = [0.0 for _ in graph.Inodes]
            scores.append(1.0)
            return array(scores, dtype=float)

        # ---- lo2 (lowest ranked neighbor) ----
        lo2_id = min(candidate_ids, key=lambda i: graph.Inodes[i].rank)
        lo2_rank = graph.Inodes[lo2_id].rank

        # ---- check opposite-side ----
        if node.node_type == 'L':
            lo2_has_opposite = len(graph.right_memory[lo2_id]) > 0
        else:
            lo2_has_opposite = len(graph.left_memory[lo2_id]) > 0

        # ---- compute scores ----
        candidate_set = set(node.candidate_Inodes)
        for inode_id in graph.Inodes:
            inode = graph.Inodes[inode_id]

            # Invalid nodes
            if not inode.available or (inode_id not in candidate_set):
                inode_scores.append(0.0)
                continue

            if lo2_has_opposite:
                # Case 2: ε = 0, only lo2 is valid
                if inode_id == lo2_id:
                    score = exp(inode.rank - 1.0)
                else:
                    score = 0.0
            else:
                # Case 3 or Case 1
                if ((node.node_type == 'L' and graph.right_memory[inode_id]) or 
                    (node.node_type == 'R' and graph.left_memory[inode_id])):
                    # eligible alternative (like lo2.5)
                    score = exp(lo2_rank - 1.0)
                else:
                    # no opposite edge → cannot match
                    score = 0.0

            inode_scores.append(score)

        # ---- WAIT handling ----
        if sum(inode_scores) == 0:
            # Case 1: no matching possible → WAIT
            inode_scores.append(1.0)
        else:
            inode_scores.append(0.0)

        scores = array(inode_scores, dtype=float)
        return scores

    def process_graph(self, graph):
        for inode in graph.Inodes.values():
            inode.rank = RND_GEN.random()

    def select_inode_for_L(self, graph, lnode):
        sorted_ids = sorted(lnode.candidate_Inodes, key=lambda id: graph.Inodes[id].rank)   

        best_available = None
        best_valid = None

        for inode_id in sorted_ids:
            inode = graph.Inodes[inode_id]
            if not inode.available:
                continue

            if best_available is None:
                best_available = inode  # Case 4

            if graph.right_memory[inode_id]:
                best_valid = inode
                break   

        if best_valid: return best_valid   # Case 2 or 3
        return None # Case 1 -> Wait

    def select_inode_for_R(self, graph, rnode):
        sorted_ids = sorted(rnode.candidate_Inodes, key=lambda id: graph.Inodes[id].rank)   

        best_available = None
        best_valid = None

        for inode_id in sorted_ids:
            inode = graph.Inodes[inode_id]
            if not inode.available:
                continue

            if best_available is None:
                best_available = inode  # Case 4

            if graph.left_memory[inode_id]:
                best_valid = inode
                break   

        if best_valid: return best_valid   # Case 2 or 3
        return None # Case 1 -> Wait

    def select_partner(self, graph, nodes):
        if(self.deterministic_partner):
            return super().select_partner(graph, nodes)

        if not nodes: return None

        nodes = tuple(nodes)
        idx = RND_GEN.integers(len(nodes))
        return nodes[idx]

from .GraphModel import TripartiteGraph