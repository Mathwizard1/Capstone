from __future__ import annotations

from abc import ABC, abstractmethod

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

from .utils import RND_GEN

# Matching Strategy Class

class MatchingStrategy(ABC):
    def __init__(self, name = "MatchingStrategy", deterministic_partner= True) -> None:
        self.name = name
        self.deterministic_partner = deterministic_partner

    def process_graph(self, graph: TripartiteGraph): 
        pass

    def select_inode_sub_optimal(self, graph: TripartiteGraph, inode_ids: tuple[int, ...]) -> INode | None:
        # Sub Optimal
        for inode_id in inode_ids:
            inode = graph.Inodes[inode_id]
            if inode.available:
                return inode
        return None

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