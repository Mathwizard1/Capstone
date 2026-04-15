from __future__ import annotations

from typing import Literal
from itertools import count

NODE_TYPE = Literal['L', 'I', 'R']

_Node_counter = count()
def _get_next_id():
    return next(_Node_counter)

from torch import Tensor

class Entity:
    def __init__(self) -> None:
        self.embedding: Tensor|None = None
        self.weight: Tensor|None = None

        self.rank = 0.0

class Node(Entity):
    def __init__(self, node_type: NODE_TYPE) -> None:
        super().__init__()
        self.id = _get_next_id()
        self.node_type = node_type

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

class varNode(Node):
    def __init__(self, online_time, candidate_Inodes: tuple[int, ...], node_type: NODE_TYPE) -> None:
        super().__init__(node_type)
        self.online_time = online_time
        self.candidate_Inodes = candidate_Inodes

        self.connected_Inode: INode | None = None

    def __str__(self) -> str:
        string = f"{self.node_type}Node(id:{self.id} online_time:{self.online_time})\n"
        return string

    def __lt__(self, other):
        if not isinstance(other, varNode):
            return NotImplemented
        return self.online_time < other.online_time
    
    def __gt__(self, other):
        if not isinstance(other, varNode):
            return NotImplemented
        return self.online_time > other.online_time

# Main Node classes

class LNode(varNode):
    def __init__(self, online_time, candidate_Inodes: tuple[int, ...]) -> None:
        super().__init__(online_time, candidate_Inodes, 'L')

class RNode(varNode):
    def __init__(self, online_time, candidate_Inodes: tuple[int, ...]) -> None:
        super().__init__(online_time, candidate_Inodes, 'R')

class INode(Node):
    def __init__(self) -> None:
        super().__init__('I')

        self.connection: tuple[LNode, RNode] | None = None
        self.available = True

    def __str__(self) -> str:
        return f"INode(id:{self.id})\n"
    
    def reset(self):
        self.connection = None
        self.available = True