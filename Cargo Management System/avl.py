from node import Node
def comp_1(node_1, node_2):
    if node_1.element[1]>node_2.element[1]:
        return True
    elif node_1.element[1]==node_2.element[1] and node_1.element[0]>node_2.element[0]:
        return True
    return False

class AVLTree:
    def __init__(self, compare_function=comp_1): #compare_function is a function that compares two nodes.
        self.root = None #compare_function should return True if the first node is greater than the second node
        self.size = 0
        self.comparator = compare_function

    def get_height(self,node):
        if not node:
            return 0
        return node.height
    
    def get_balance(self,node):
        if not node:
            return 0
        return self.get_height(node.left)-self.get_height(node.right)
    
    def balance_tree(self, node):
        balance = self.get_balance(node)

        # Left Left Case
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.rotate_right(node)

        # Left Right Case
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)

        # Right Right Case
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.rotate_left(node)

        # Right Left Case
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node
    
    def rotate_left(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))

        return y

    def rotate_right(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))

        return y
    
    def insert(self,node):
        self.root=self._insert(node,self.root)
        self.size+=1
    
    def _insert(self,node,root):
        if not root:
            return node
        if self.comparator(node,root):
            root.right=self._insert(node,root.right)
        elif self.comparator(root,node):
            root.left=self._insert(node,root.left)
        root.height=1+max(self.get_height(root.left),self.get_height(root.right))
        return self.balance_tree(root)
    
    def inorder(self, node):
        arr = []
        self._inorder(node, arr)
        return arr

    def _inorder(self, node, arr): #left, root, right
        if not node:
            return
        self._inorder(node.left, arr)
        arr.append(node.element[0])
        self._inorder(node.right, arr)

    def delete(self,node):
        self.root=self._delete(node,self.root)
        self.size-=1

    def _delete(self,node,root):
        if not root:
            return root
        if self.comparator(node,root):
            root.right=self._delete(node,root.right)
        elif self.comparator(root,node):
            root.left=self._delete(node,root.left)
        else:
            if not root.right:
                return root.left
            elif not root.left:
                return root.right
            else:
                temp = self.get_min_value_node(root.right)
                root.element = temp.element
                root.capacity=temp.capacity
                root.right = self._delete(temp, root.right)
        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        return self.balance_tree(root)

    def get_min_value_node(self, node):
        if node is None or node.left is None:
            return node
        return self.get_min_value_node(node.left) #keep going left to find the minimum
    
    def get_max_value_node(self, node):
        if node is None or node.right is None:
            return node
        return self.get_max_value_node(node.right) #keep going right to find the maximum
    
    def find(self, node):
        return self._find(node, self.root)

    def _find(self, node, root):
        if not root:
            return None  # Node not found
        if self.comparator(node, root):
            return self._find(node, root.right)
        elif self.comparator(root, node):
            return self._find(node, root.left)
        else:
            return root

    def compactfit_minid(self, node): #finds the node with the minimum size abd minimum id that fits the given object
        r=self.root #start searching from the root
        n = self.get_max_value_node(self.root)
        target = None  
        if node.element[1] > n.element[1]:
            return None
        while r is not None:
            if node.element[1] <= r.element[1]:
                target = r  
                r = r.left  
            else:
                r = r.right 
        return target

    def compactfit_maxid(self,node):
        n = self.get_max_value_node(self.root)
        if node.element[1] > n.element[1]:
            return None
        ans=self.compactfit_minid(node) #we first find the minimum size node that fits the object
        s=ans.element[1]
        r=self.root
        target=None #after the loop, target will be the node with the maximum id that fits
        while r is not None:
            if r.element[1]>s:
                r=r.left
            elif r.element[1]<s:
                r=r.right
            else:
                target=r
                r=r.right
        return target
    
    def largestfit_maxid(self,node):
        n=self.get_max_value_node(self.root) #just returns the rightmost node if it fits the object
        if node.element[1]>n.element[1]:
            return None
        return n
    
    def largestfit_minid(self,node): #checks the minimum id for the largest size that fits the object
        n = self.get_max_value_node(self.root)
        if node.element[1]>n.element[1]:
            return None
        s=n.element[1]
        target=None
        r=self.root
        while r is not None:
            if r.element[1]<s:
                r=r.right
            elif r.element[1]==s:
                target=r
                r=r.left
        return target
