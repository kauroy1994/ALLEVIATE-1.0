from copy import deepcopy
class Node(object):
    '''search node in the search tree
    '''

    def __init__(self,value = None,pos="root",depth=0,parent="root"):
        '''initialize node attributes
        '''
        
        self.value = value
        self.left = None
        self.right = None
        self.pos = pos
        self.depth = depth
        self.parent = parent

    def get_id(self):
        '''get id label for node'''

        x = deepcopy(self)
        iden = []
        while True:
            iden += [x.pos]
            if x.parent == "root":
                break
            x = x.parent

        return "_".join(iden)

    def __str__(self):
        return self.value
        

class Search_tree(object):
    '''creates search tree'''

    costs = {}
    nodes = []
    phrases = ['Do you feel','How often do you feel','Any ideas as to what is causing your','Have you tried rememdies to fix your','Do you feel any other symptoms besides']
    max_depth = len(phrases)

    def __init__(self,x=None):
        
        self.x = x
        self.stack = [Node(value = Search_tree.phrases[0]+' '+self.x)]
        self.tree = []
        self.dot_file = "digraph G{"+'\n'

    def create_search_tree(self):
        
        while (self.stack):
            node = self.stack.pop()
            if node.depth == Search_tree.max_depth-1:
                continue
            node.left = Node(Search_tree.phrases[node.depth+1]+' '+self.x,pos="left",depth=node.depth+1,parent=node)
            node.right = Node(Search_tree.phrases[node.depth+1]+' '+self.x,pos="right",depth=node.depth+1,parent=node)
            self.stack.append(node.right)
            self.stack.append(node.left)
            self.tree.append(node)

    def traverse_search_tree(self):

        tree_copy = deepcopy(self.tree)
        stack = [tree_copy[0]]
        while (stack):
            node = stack.pop()
            if node.depth == Search_tree.max_depth-1:
                continue
            if node.pos == "root":
                node_id = node.get_id()
                self.dot_file += node_id+"[label="+'\"'+str(node)+'\"'+']'+'\n'
            elif node.pos == "left":
                node_id = node.get_id()
                self.dot_file += node_id+"[label="+'\"'+str(node)+'\"'+']'+'\n'
                self.dot_file += node.parent.get_id()+"->"+node_id+"[label=1]"+'\n'
            elif node.pos == "right":
                node_id = node.get_id()
                self.dot_file += node_id+"[label="+'\"'+str(node)+'\"'+']'+'\n'
                self.dot_file += node.parent.get_id()+"->"+node_id+"[label=0]"+'\n'
            stack.append(node.right)
            stack.append(node.left)
        self.dot_file += '}'
            
'''
tree = Search_tree(x="nervous")
tree.create_search_tree()
tree.traverse_search_tree()
print (tree.dot_file)
'''
