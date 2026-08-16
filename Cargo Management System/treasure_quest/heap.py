'''
Python Code to implement a heap with general comparison function
'''
def floor(num):
    return int(num-num%1)

def compfn(x,y):
    return x<y

class Heap:
    '''
    Class to implement a heap with general comparison function
    '''
    
    def __init__(self, comparison_function, init_array):
        '''
        Arguments:
            comparison_function : function : A function that takes in two arguments and returns a boolean value
            init_array : List[Any] : The initial array to be inserted into the heap
        Returns:
            None
        Description:
            Initializes a heap with a comparison function
            Details of Comparison Function:
                The comparison function should take in two arguments and return a boolean value
                If the comparison function returns True, it means that the first argument is to be considered smaller than the second argument
                If the comparison function returns False, it means that the first argument is to be considered greater than or equal to the second argument
        Time Complexity:
            O(n) where n is the number of elements in init_array
        '''
        
        # Write your code here
        self.heap_arr=[]
        self.comp_fn=comparison_function #comp_fn(x,y) -> x<y
        self.heap_size=len(self.heap_arr)
        for elem in init_array:
            self.insert(elem)
        
    def insert(self, value):
        '''
        Arguments:
            value : Any : The value to be inserted into the heap
        Returns:
            None
        Description:
            Inserts a value into the heap
        Time Complexity:
            O(log(n)) where n is the number of elements currently in the heap
        '''
        
        # Write your code here
        self.heap_arr.append(value)
        self.heap_size+=1
        self.upheap()
        pass
    
    def extract(self):
        '''
        Arguments:
            None
        Returns:
            Any : The value extracted from the top of heap
        Description:
            Extracts the value from the top of heap, i.e. removes it from heap
        Time Complexity:
            O(log(n)) where n is the number of elements currently in the heap
        '''
        
        # Write your code here
        top_ele=self.heap_arr[0]
        self.heap_arr[0]=self.heap_arr[-1]
        self.heap_arr.pop()
        self.heap_size-=1
        self.downheap()
        return top_ele
    
    def top(self):
        '''
        Arguments:
            None
        Returns:
            Any : The value at the top of heap
        Description:
            Returns the value at the top of heap
        Time Complexity:
            O(1)
        '''
        
        # Write your code here
        return self.heap_arr[0]
    
    # You can add more functions if you want to
    
    #a function to perform upheap
    def _upheap(self,ind):
        parent_ind=floor((ind-1)/2) #the index of the parent of position ind
        if self.comp_fn(self.heap_arr[ind],self.heap_arr[parent_ind]):
            self.heap_arr[ind],self.heap_arr[parent_ind]=self.heap_arr[parent_ind],self.heap_arr[ind]
            if parent_ind>0:
                return self._upheap(parent_ind)
    
    #simple function call for upheap        
    def upheap(self):
        return self._upheap(self.heap_size-1)
    
    #a function to perform downheap
    def _downheap(self,ind):
        left_ind=2*ind+1
        right_ind=2*ind+2
        if left_ind>=self.heap_size:
            return
        elif left_ind==self.heap_size-1:
            if self.comp_fn(self.heap_arr[left_ind],self.heap_arr[ind]):
                self.heap_arr[ind],self.heap_arr[left_ind]=self.heap_arr[left_ind],self.heap_arr[ind]
        else:
            if self.comp_fn(self.heap_arr[left_ind],self.heap_arr[right_ind]):
                min_ind=left_ind
            else: min_ind=right_ind
            if self.comp_fn(self.heap_arr[min_ind],self.heap_arr[ind]):
                self.heap_arr[min_ind],self.heap_arr[ind]=self.heap_arr[ind],self.heap_arr[min_ind]
                return self._downheap(min_ind)
    
    #simple function call for downheap
    def downheap(self):
        return self._downheap(0)

if __name__=="__main__":
    #print(floor(1.99))
    newheap=Heap(compfn,[3,2,1,4,5,6,10,3.1,0.5])
    print(newheap.heap_arr)
    newheap.extract()
    print(newheap.heap_arr)