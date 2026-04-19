from __future__ import annotations

from typing import overload, Literal

from .Nodes import (
    Entity,
    NODE_TYPE, varNode,
    LNode, INode, RNode
)

# Graph Class

class TripartiteGraph(Entity):
    def __init__(self, strategy: MatchingStrategy, n_Inodes:int = 1) -> None:
        super().__init__()
        self.strategy = strategy

        self.left_memory: dict[int, set[varNode]] = {}
        self.right_memory: dict[int, set[varNode]] = {}
        self.matches = 0

        self.Inodes: dict[int, INode] = {}

        for _ in range(n_Inodes):
            inode = INode()
            
            self.Inodes[inode.id] = inode
            self.left_memory[inode.id] = set()
            self.right_memory[inode.id] = set()

        self.strategy.process_graph(self)

    def __str__(self) -> str:
        string = "TripartiteGraph(\n"
        for Inode in self.Inodes.values(): string += str(Inode) + "\n"
        return string + ")"

    @overload # Overload for 'L'
    def add_node(self, online_time, candidate_Inodes, node_type: Literal['L']) -> LNode: ...
    @overload # Overload for 'R'
    def add_node(self, online_time, candidate_Inodes, node_type: Literal['R']) -> RNode: ...

    def add_node(self, online_time, candidate_Inodes, node_type: NODE_TYPE):
        node = LNode(online_time, candidate_Inodes) if(node_type == 'L') else RNode(online_time, candidate_Inodes)
        for inode_id in candidate_Inodes:
            if(not self.Inodes[inode_id].available): continue

            if(node_type == 'L'):
                self.left_memory[inode_id].add(node)
            else:
                self.right_memory[inode_id].add(node)
        return node

    def add_Lnode(self, online_time, candidate_Inodes, discard_node= False):
        node = self.add_node(online_time, candidate_Inodes, "L")
        self.process_Lnode(node, discard_node)
        return node.id

    def add_Rnode(self, online_time, candidate_Inodes, discard_node= False):
        node = self.add_node(online_time, candidate_Inodes, "R")
        self.process_Rnode(node, discard_node)
        return node.id

    def process_Lnode(self, lnode: LNode, discard_node):
        inode = self.strategy.select_inode_for_L(self, lnode)

        if inode:
            partner = self.strategy.select_partner(self, self.right_memory[inode.id])
            if partner: 
                self.match(lnode, inode, partner) # type: ignore
        elif discard_node:
            for inode_id in lnode.candidate_Inodes:
                if(inode_id in self.left_memory):
                    self.left_memory[inode_id].discard(lnode)

    def process_Rnode(self, rnode: RNode, discard_node):
        inode = self.strategy.select_inode_for_R(self, rnode)

        if inode:
            # At least a pair has been made
            inode.waiting()

            partner = self.strategy.select_partner(self, self.left_memory[inode.id])
            if partner: 
                self.match(partner, inode, rnode) # type: ignore
        elif discard_node:
            for inode_id in rnode.candidate_Inodes:
                if(inode_id in self.right_memory):
                    self.right_memory[inode_id].discard(rnode)

    def match(self, lnode: LNode, inode: INode, rnode: RNode):
        lnode.connected_Inode = inode
        rnode.connected_Inode = inode

        inode.connection = (lnode, rnode)
        inode.offline()

        self.matches += 1

        # remove memory for this inode
        self.left_memory.pop(inode.id, None)
        self.right_memory.pop(inode.id, None)

        # remove memory for lnode and rnode
        for inode_id in self.Inodes:
            if(inode_id in self.left_memory):
                self.left_memory[inode_id].discard(lnode)
            if(inode_id in self.right_memory):
                self.right_memory[inode_id].discard(rnode)

        #print("MATCH:", lnode, "→", inode, "→", rnode)

    def compute_competitive_ratio(self, opt):
        return self.matches / opt

    def reset(self):
        # Reset matching stats
        self.matches = 0    

        # Reset memory
        self.left_memory.clear()
        self.right_memory.clear()

        # Reset inode state and memory
        for inode in self.Inodes.values():
            inode.reset()

            self.left_memory[inode.id] = set()
            self.right_memory[inode.id] = set() 

        # Reinitialize strategy-specific state
        self.strategy.reset(self)

    def get_state(self, node: varNode):
        inode_features = []
        edge_features = []

        candidate_set = set(node.candidate_Inodes)
        for inode_id, inode in self.Inodes.items():
            left_cnt = len(self.left_memory[inode_id]) if inode.available else -1.0
            right_cnt = len(self.right_memory[inode_id]) if inode.available else -1.0

            # ---- INODE FEATURES (global order-invariant) ----
            inode_features.append([
                float(inode.available),
                float(inode.state),
                float(left_cnt),
                float(right_cnt),
                float(left_cnt - right_cnt),   # imbalance
                float(left_cnt + right_cnt),   # congestion
            ])

            # ---- EDGE FEATURES ----
            edge_features.append([
                float(0.0 if node.node_type == 'L' else 1.0),
                float(node.online_time),
                float(len(node.candidate_Inodes)),
                float(1.0 if inode_id in candidate_set else 0.0),
            ])

        # ---- GLOBAL FEATURES ----
        total_left = sum(len(v) for v in self.left_memory.values())
        total_right = sum(len(v) for v in self.right_memory.values())

        global_features = [
            float(self.matches),
            float(total_left),
            float(total_right),
            float(total_left - total_right),  # global imbalance
        ]

        return {
            "inode": inode_features,
            "edge": edge_features,
            "global": global_features
        }

from .GraphStrategy import MatchingStrategy