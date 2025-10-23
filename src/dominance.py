from typing import Dict, Set

from cfg import CFG, BasicBlock


class DominatorTree:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.dom: Dict[BasicBlock, Set[BasicBlock]] = {}
        self.idom: Dict[BasicBlock, BasicBlock] = {}
        self.dom_frontiers: Dict[BasicBlock, Set[BasicBlock]] = {}

    def compute_dominators(self) -> None:
        return

    def compute_idom(self) -> None:
        return

    def compute_dominance_frontiers(self) -> None:
        return
