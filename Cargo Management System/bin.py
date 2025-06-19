from avl import AVLTree
from node import Node
from object import Object , Color
def compare(node1,node2):
    if node1.element[0]>node2.element[0]:
        return True
    return False
class Bin:
    def __init__(self, bin_id, capacity): # Initialize a Bin with an ID and capacity
        self.root=None
        self.bin_id=bin_id
        self.capacity=capacity
        self.objects=AVLTree(compare) #objects are stored in order of ID in the AVL tree
        pass

    def add_object(self, object): 
        # Implement logic to add an object to this bin
        self.capacity-=object.size
        node=Node(object.object_id,object.size)
        self.objects.insert(node)
        pass

    def remove_object(self, object_id):
        # Implement logic to remove an object by ID
        node=Node(object_id,0)
        delete_size=self.objects.find(node).element[1]
        self.objects.delete(node)
        self.capacity=self.capacity+delete_size
        pass
