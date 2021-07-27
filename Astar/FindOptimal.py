from copy import deepcopy
from SearchTree import Search_tree
from random import random

def main():
    stree = Search_tree(x="nervous")
    stree.create_search_tree()
    tree = stree.tree
    node_candidates = deepcopy(tree)
    path = []
    
    while node_candidates:
        node_to_expand = node_candidates[0]
        for node in node_candidates:
            if node.cost + node.heuristic >= node_to_expand.cost + node_to_expand.heuristic:
                node_to_expand = node
        
        path.append(str(node_to_expand))
        answered = int(random() < 0.8)
        path.append(answered)
        if node_to_expand.left == None or node_to_expand.right == None:
            break
        node_candidates = []
        if answered:
            node_candidates.append(node_to_expand.left)
        else:
            break
            node_candidates.append(node_to_expand.right)

    return path

print(main())
                

