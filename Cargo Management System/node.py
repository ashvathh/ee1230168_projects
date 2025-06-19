class Node:
    def __init__(self,id,size):
        self.left=None
        self.right=None
        self.element=[id,size]
        self.height=1
        self.bin=None
        self.capacity=size
        pass
