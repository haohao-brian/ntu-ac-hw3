from typing import Dict, List, Set
from bril import Function, Instruction

class BasicBlock:
    def __init__(self, label: str):
        self.label = label
        self.instructions: List[Instruction] = []
        self.predecessors: Set['BasicBlock'] = set()
        self.successors: Set['BasicBlock'] = set()

    def __repr__(self):
        return f'BasicBlock({self.label})'

class CFG:
    def __init__(self, function: Function):
        self.function = function
        self.blocks: Dict[str, BasicBlock] = {}
        self.entry_block: BasicBlock = self.construct_cfg()

    def construct_cfg(self) -> BasicBlock:
        """
        Constructs the CFG for the function and returns the entry block.
        """
        # TODO: Implement CFG construction logic
        # Steps:
        # 1. Divide instructions into basic blocks.
        # 2. Establish successor and predecessor relationships.
        # 3. Handle labels and control flow instructions.
        pass

    def get_blocks(self) -> List[BasicBlock]:
        return list(self.blocks.values())