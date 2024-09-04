import re
from dataclasses import dataclass
from types import SimpleNamespace
if __name__ == "__main__":
    from utils import Taxa
else:
    from .utils import Taxa

import warnings

with warnings.catch_warnings(action="ignore"):
    import ete3

JPLACE_FIXER = re.compile(r"\{[0-9]+\}")


def make_leaf_set(n: ete3.TreeNode) -> set[str]:
    return {leaf.name for leaf in n}


def assign_jplace_ids(tree: ete3.Tree):
    for i, n in enumerate(tree.traverse("postorder")):
        n.add_features(jplace_id=i)


def assign_expected_placement_ids(tree: ete3.Tree, removed_taxa: set[str]):
    index = 0
    for n in tree.traverse("postorder"):
        cur_taxa = {leaf.name for leaf in n}
        if cur_taxa.issubset(removed_taxa):
            continue
        n.add_features(jplace_id=index)
        index += 1


def find_true_node(tree: ete3.Tree, taxa: set[str]):
    if len(taxa) == 1:
        (taxa_name, ) = taxa
        node = tree & taxa_name
    else:
        node = tree.get_common_ancestor(taxa)
    node_list = [node]
    while not hasattr(node, "jplace_id"):
        node = node.up
        node_list.append(node)
    child, last = node_list[-2:]
    if last.children[0] == child:
        return node.children[1]
    return node.children[0]


def find_node_by_jplace_id(tree: ete3.Tree, jplace_id: int):
    for node in tree.traverse("postorder"):
        if hasattr(node, "jplace_id") and node.jplace_id == jplace_id:
            return node


@dataclass
class PlacementElement:
    taxa: Taxa
    placements: list[dict]
    nd = float("nan")
    e_nd = float("nan")

    def json(self):
        return {"nd": self.nd, "e_nd": self.e_nd, "formatted_name":
                self.taxa.formatted_name()} | self.taxa.json()


class Jplace:
    def __init__(self, json, true_tree: ete3.Tree):
        self._parse_tree(json["tree"])
        self._parse_placements(json["placements"], json["fields"])
        self._true_tree = true_tree
        self._fix_tree()

    def set_true_tree(self, true_tree: ete3.Tree, pruned_leaves: set[str]):
        self._true_tree = true_tree
        self._fix_tree()
        assign_expected_placement_ids(self._true_tree, pruned_leaves)

    @property
    def placements(self) -> [PlacementElement]:
        return self._placements

    def _parse_tree(self, tree: str):
        self._tree = ete3.Tree(JPLACE_FIXER.sub("", tree))
        assign_jplace_ids(self._tree)

    def _fix_tree(self):
        for n in self._tree.traverse('postorder'):
            if len(n.children) == 0:
                continue
            pl_child = make_leaf_set(n.children[0])
            pr_child = make_leaf_set(n.children[1])
            children = pl_child | pr_child
            tn = self._true_tree.get_common_ancestor(children)

            tl_child = make_leaf_set(tn.children[0])

            if not pl_child.issubset(tl_child):
                tn.children[0], tn.children[1] = tn.children[1], tn.children[0]

    @staticmethod
    def _make_placement_row(placement, fields):
        obj = SimpleNamespace()

        for n, v in zip(fields, placement):
            setattr(obj, n, v)
        return obj

    def _parse_placements(self, placements, fields):
        self._placements = []
        for p in placements:
            node_name = p["n"][0]
            row = [Jplace._make_placement_row(k, fields) for k in p["p"]]
            self._placements.append(
                PlacementElement(Taxa.parse(node_name), row))

    @staticmethod
    def _compute_nd(tree: ete3.Tree, taxa: Taxa, edge_num: int) -> float:
        taxa_label_set = {taxa.name}
        true_node = find_true_node(tree, taxa_label_set)
        placed_node = find_node_by_jplace_id(tree, edge_num)
        return true_node.get_distance(placed_node, topology_only=True)

    @staticmethod
    def _compute_end_for_placement(
        tree: ete3.Tree, placement: PlacementElement
    ) -> float:
        total = 0.0
        total_lwr = 0.0
        for p in placement.placements:
            total += (
                Jplace._compute_nd(tree, placement.taxa, p.edge_num)
                * p.like_weight_ratio
            )
            total_lwr += p.like_weight_ratio

        return total / total_lwr

    def compute_nds(self):
        for p in self._placements:
            p.nd = Jplace._compute_nd(
                self._true_tree, p.taxa, p.placements[0].edge_num)
            p.e_nd = Jplace._compute_end_for_placement(self._true_tree, p)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--jplace", type=str)
    parser.add_argument("--tree", type=str)
    parser.add_argument("--pruned", type=str)
    args = parser.parse_args()

    true_tree = ete3.Tree(open(args.tree).read())
    removed_taxa = json.load(open(args.pruned))['pruned_leaves']
    assign_expected_placement_ids(
        true_tree, set(removed_taxa))

    breakpoint()
    jp = Jplace(json.load(open(args.jplace)), true_tree)
    jp.compute_nds()
