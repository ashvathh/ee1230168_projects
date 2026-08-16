'''
    Python file to implement the class CrewMate
'''
from heap import Heap
def priority_comp_fn(x,y): #current time is the time of insertion of the object
    '''current_time=0#since it doesnt make a difference in comparator as it its used in both
    p1=(current_time-x.arrival_time-x.size,x.id)
    p2=(current_time-y.arrival_time-y.size,y.id)
    return p1>p2 #since we want maxheap based on priority'''
    return (x.size+x.arrival_time)<(y.size+y.arrival_time)



class CrewMate:
    '''
    Class to implement a crewmate
    '''
    
    def __init__(self,id):
        '''
        Arguments:
            None
        Returns:
            None
        Description:
            Initializes the crewmate
        '''
        
        # Write your code here
        self.member_id=id
        self.treasure_arr=[] #stores the instances of Treasure()
        self.treasure_info_arr=[] #stores info of treasures
        #self.treasure_heap=Heap(priority_comp_fn,[])
        self.working_time=0
        self.heap_start_time=0
        self.treasure_count=0
        self.treasure_completion=[]
        '''
        self.member_id=id
        self.treasure_heap=Heap(priority_comp_fn,[])
        self.treasure_arr=[]
        self.treasure_id_arr=[]
        self.load=0
        self.current_treasure=None
        self.working_time=0 # the total time that a crew member works for
        self.heap_start_time=0
        self.current_time=0
        pass'''
    
    # Add more methods if required
    def add_treasure(self,treasure): #treasue = treasure instance
        self.treasure_count+=1
        self.treasure_arr.append(treasure)
        self.treasure_info_arr.append([treasure.id,treasure.size,treasure.arrival_time])
        if self.working_time<=treasure.arrival_time:
            self.working_time=treasure.arrival_time+treasure.size
        else:
            self.working_time+=treasure.size
        print(self.working_time)

    def get_completion_time_for_member(self):
        self.treasure_completion=[]
        self.treasure_heap=Heap(priority_comp_fn,[])
        self.heap_start_time=0


        def add_treasure_to_heap1(treasure1):
            current_treasure=treasure1
            if self.treasure_heap.heap_arr==[]:
                self.treasure_heap.insert(current_treasure)
                self.heap_start_time=current_treasure.arrival_time
                #print(self.heap_start_time,end=' ')
            else:
                if self.working_time<=treasure1.arrival_time:
                    while self.treasure_heap.heap_arr!=[]:
                        removed_ele=self.treasure_heap.extract()
                        self.heap_start_time+=removed_ele.size
                        #print(self.heap_start_time,end=' ')
                        removed_ele.completion_time=self.heap_start_time
                        self.treasure_completion.append(removed_ele)
                    if self.treasure_heap.heap_arr==[]:
                        self.treasure_heap.insert(current_treasure)
                        self.heap_start_time=current_treasure.arrival_time
                else:
                    if self.heap_start_time<=treasure1.arrival_time:
                        while self.treasure_heap.heap_arr!=[]:
                            top_ele=self.treasure_heap.top()
                            if top_ele.size+self.heap_start_time<=treasure1.arrival_time:
                                removed_ele=self.treasure_heap.extract()
                                self.heap_start_time+=removed_ele.size
                                #print(self.heap_start_time,end=' ')
                                removed_ele.completion_time=self.heap_start_time
                                self.treasure_completion.append(removed_ele)
                            else: break
                    if self.treasure_heap.heap_arr==[]:
                        self.treasure_heap.insert(current_treasure)
                        self.heap_start_time=current_treasure.arrival_time
                    else:
                        top_treasure=self.treasure_heap.top()
                        if self.heap_start_time!=treasure1.arrival_time:
                            top_treasure.size-=(treasure1.arrival_time-self.heap_start_time)
                        self.heap_start_time=treasure1.arrival_time
                        self.treasure_heap.insert(treasure1)

        for ele in self.treasure_arr:
            add_treasure_to_heap1(ele)
        def get_completion_time_for_member1():
            #self.copy1=self.treasure_heap
            #print(type(self.copy1))
            while self.treasure_heap.heap_arr!=[]:
                removed_ele=self.treasure_heap.extract()
                self.heap_start_time+=removed_ele.size
                removed_ele.completion_time=self.heap_start_time
                self.treasure_completion.append(removed_ele)
            
            w=self.working_time
            if self.treasure_completion!=[]:
                maxt=self.treasure_completion[0].completion_time
                for ele in self.treasure_completion:
                    if ele.completion_time>maxt:
                        maxt=ele.completion_time
                
                for ele in self.treasure_completion:
                    ele.completion_time+=(w-maxt)

            
            return self.treasure_completion
        return get_completion_time_for_member1()
                


    '''
        self.treasure_arr.append(treasure)
        self.treasure_id_arr.append(treasure.id)
        self.current_time=treasure.arrival_time
        if treasure.arrival_time>=self.heap_start_time+self.load:
            self.heap_start_time=treasure.arrival_time
            self.load=treasure.size
        else:
            self.load+=treasure.size
    '''



    '''
        self.load+=treasure.size
        self.treasure_arr.append((treasure.id,treasure.size,treasure.arrival_time,treasure.completion_time)) # change 0 to completion time later on

        #!!! CHECK ADDITION TO HEAP FUNCTION
        self.treasure_heap.insert(treasure)#check this!!!
        #!!!

        if self.working_time<=treasure.arrival_time:
            self.working_time=treasure.size+treasure.arrival_time # if heap was empty when the new object was assigned to crew member
        else:
            self.working_time+=treasure.size #the treasure need not be processed as soon as it is inserted, but all of it will get processed one after the other.
    '''

if __name__=="__main__":
    cr1=CrewMate()