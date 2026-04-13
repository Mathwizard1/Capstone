from __future__ import annotations

from abc import ABC, abstractmethod

from .Nodes import (
    varNode,
    LNode, INode, RNode
)

from .utils import RND_GEN

# Matching Strategy Class

class MatchingStrategy(ABC):
    def __init__(self, name = "MatchingStrategy") -> None:
        self.name = name

    def process_graph(self, graph: TripartiteGraph): pass

    @abstractmethod 
    def select_inode_sub_optimal(self, graph: TripartiteGraph, node: varNode) -> INode | None: pass

    @abstractmethod
    def select_inode_for_L(self, graph: TripartiteGraph, lnode: LNode) -> INode | None: pass

    @abstractmethod
    def select_inode_for_R(self, graph: TripartiteGraph, rnode: RNode) -> INode | None: pass

    def select_partner(self, graph: TripartiteGraph, nodes: set[varNode]):
        if not nodes:
            return None
        return next(iter(nodes))

class GreedyStrategy(MatchingStrategy):
    def __init__(self, name= "GreedyStrategy") -> None:
        super().__init__(name)
    
    def select_inode_sub_optimal(self, graph, node) -> INode | None:
        # Sub Optimal
        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available:
                return inode
        return None

    def select_inode_for_L(self, graph, lnode):
        # Optimal with R connected
        for inode_id in lnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available and graph.right_memory[inode_id]:
                return inode
            
        return self.select_inode_sub_optimal(graph, lnode)

    def select_inode_for_R(self, graph, rnode):
        # Optimal with L connected
        for inode_id in rnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available and graph.left_memory[inode_id]:
                return inode
            
        return self.select_inode_sub_optimal(graph, rnode)

class RankStrategy(MatchingStrategy):
    def __init__(self, name= "RankStrategy") -> None:
        super().__init__(name)

    def process_graph(self, graph):
        for inode in graph.Inodes.values():
            inode.rank = RND_GEN.random()

    def select_inode_sub_optimal(self, graph, node) -> INode | None:
        best = None
        best_rank = float('inf')

        for inode_id in node.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available and inode.rank < best_rank:
                best_rank = inode.rank
                best = inode
        return best

    def select_inode_for_L(self, graph, lnode):
        best = None
        best_rank = float('inf')

        for inode_id in lnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available and graph.right_memory[inode_id]:
                if inode.rank < best_rank:
                    best_rank = inode.rank
                    best = inode

        if best: return best
        return self.select_inode_sub_optimal(graph, lnode)

    def select_inode_for_R(self, graph, rnode):
        best = None
        best_rank = float('inf')

        for inode_id in rnode.candidate_Inodes:
            inode = graph.Inodes[inode_id]
            if inode.available and graph.left_memory[inode_id]:
                if inode.rank < best_rank:
                    best_rank = inode.rank
                    best = inode

        if best: return best
        return self.select_inode_sub_optimal(graph, rnode)

from .GraphModel import TripartiteGraph