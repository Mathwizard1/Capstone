from __future__ import annotations

from typing import Literal
from itertools import count

from enum import IntEnum

from torch import Tensor

NODE_TYPE = Literal['L', 'I', 'R']

_Node_counter = count()
def _get_next_id():
    return next(_Node_counter)

class Entity:
    def __init__(self) -> None:
        self.embedding: Tensor = None # type: ignore
        self.weight: Tensor= None # type: ignore

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
    class INode_state(IntEnum):
        Offline = 0
        Waiting = 1
        Online = 2

    def __init__(self) -> None:
        super().__init__('I')

        self.connection: tuple[LNode, RNode] | None = None
        self._available: INode.INode_state = self.INode_state.Online

    @property
    def state(self):
        return self._available.value

    @property
    def available(self) -> bool:
        return bool(self._available.value)

    def waiting(self):
        self._available = self.INode_state.Waiting

    def offline(self):
        self._available = self.INode_state.Offline

    def __str__(self) -> str:
        return f"INode(id:{self.id})\n"

    def reset(self):
        self.connection = None
        self._available: INode.INode_state = self.INode_state.Online