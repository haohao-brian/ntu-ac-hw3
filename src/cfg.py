from typing import Dict, List

from bril import Function, Instruction


class BasicBlock:
    def __init__(self, label: str):
        self.label = label
        self.instructions: List[Instruction] = []

    def __repr__(self) -> str:
        return f"BasicBlock({self.label})"


class CFG:
    def __init__(self, function: Function):
        self.function = function
        self.blocks: Dict[str, BasicBlock] = {}
        self.entry_block = BasicBlock("entry")

    def get_blocks(self) -> List[BasicBlock]:
        return list(self.blocks.values())
