from bin import Bin
from avl import AVLTree
from object import Object, Color
from exceptions import NoBinFoundException
from node import Node
def compare(node1,node2):
    if node1.element[0]>node2.element[0]:
        return True
    return False
class GCMS:
    def __init__(self):
        # Maintain all the Bins and Objects in GCMS
        self.tree=AVLTree() # This AVL tree just stores bins in order of their ID then capacity. Used for finding bins that can fit objects
        self.objects=AVLTree(compare) # This AVL tree will store objects in order of their ID
        self.bin_tree=AVLTree(compare) # This AVL tree will store actual bin instances in order of their ID
        pass 

    def add_bin(self, bin_id, capacity):
        self.tree.insert(Node(bin_id,capacity))
        n=Node(bin_id,capacity)
        n.bin=Bin(bin_id,capacity)
        self.bin_tree.insert(n)
        pass

    def add_object(self, object_id, size, color):
        if color.value==1:
            n=self.tree.compactfit_minid(Node(object_id,size)) #finds the bin with the minimum size and minimum id that fits the given object
            if n is None:
                raise NoBinFoundException
            o=Node(object_id,n.element[0])
            o.capacity=size
            self.objects.insert(o) #insert the object in the objects AVL tree
            b_id=n.element[0]
            b_size=n.element[1]
            self.tree.delete(n) #delete the bin from the bins AVL tree
            #update the bin size
            b_size-=size
            self.tree.insert(Node(b_id,b_size))  #add the bin back to the bins AVL tree with updated size
            #update the bin in the bin_tree AVL tree
            m=self.bin_tree.find(Node(b_id,0))
            m.capacity-=size
            m.bin.add_object(Object(object_id,size,color))
        
        elif color.value==2:
            n=self.tree.compactfit_maxid(Node(object_id,size))#finds the bin with the minimum size and maximum id that fits the given object
            if n is None:
                raise NoBinFoundException
            o=Node(object_id,n.element[0])
            o.capacity=size
            self.objects.insert(o)
            b_id=n.element[0]
            b_size=n.element[1]
            self.tree.delete(n)
            b_size-=size
            self.tree.insert(Node(b_id,b_size))
            m=self.bin_tree.find(Node(b_id,0))
            m.capacity-=size
            m.bin.add_object(Object(object_id,size,color))

        elif color.value==3:
            n=self.tree.largestfit_minid(Node(object_id,size))#finds the bin with the maximum size and minimum id that fits the given object
            if n is None:
                raise NoBinFoundException
            o=Node(object_id,n.element[0])
            o.capacity=size
            self.objects.insert(o)
            b_id=n.element[0]
            b_size=n.element[1]
            self.tree.delete(n)
            b_size-=size
            self.tree.insert(Node(b_id,b_size))
            m=self.bin_tree.find(Node(b_id,0))
            m.capacity-=size
            m.bin.add_object(Object(object_id,size,color))

        elif color.value==4:
            n=self.tree.largestfit_maxid(Node(object_id,size))#finds the bin with the maximum size and maximum id that fits the given object
            if n is None:
                raise NoBinFoundException
            o=Node(object_id,n.element[0])
            o.capacity=size
            self.objects.insert(o)
            b_id=n.element[0]
            b_size=n.element[1]
            self.tree.delete(n)
            b_size-=size
            self.tree.insert(Node(b_id,b_size))
            m=self.bin_tree.find(Node(b_id,0))
            m.capacity-=size
            m.bin.add_object(Object(object_id,size,color))

    def delete_object(self, object_id):
        # Implemented logic to remove an object from its bin
        n=self.objects.find(Node(object_id,0)) # find the object in the objects AVL tree
        if n is None:
            return None
        size=n.capacity
        bin_id=n.element[1]
        m=self.bin_tree.find(Node(bin_id,0))# find the bin in the bin_tree AVL tree which contains the object
        binsize=m.capacity
        t=self.tree.find(Node(bin_id,binsize))#find the bin in the "tree" AVL tree which contains the object
        self.tree.delete(t)# delete the bin from the "tree" AVL tree
        # update the bin size
        self.tree.insert(Node(bin_id,binsize+size))#insert the bin back to the "tree" AVL tree with updated size
        #update the actual bin instance in the bin_tree AVL tree
        m.capacity+=size
        m.bin.remove_object(object_id)
        self.objects.delete(Node(object_id,bin_id))
        pass

    def bin_info(self, bin_id):
        # returns a tuple with current capacity of the bin and the list of objects in the bin (int, list[int])
        n=self.bin_tree.find(Node(bin_id,0))
        if n is None:
            return None
        curr_capacity=n.capacity
        l=n.bin.objects.inorder(n.bin.objects.root)
        return (curr_capacity,l)

    def object_info(self, object_id):
        # returns the bin_id in which the object is stored
        n=self.objects.find(Node(object_id,0))
        if n is None:
            return None
        return n.element[1]
