'''
    This file contains the class definition for the StrawHat class.
'''

from crewmate import CrewMate
from heap import Heap
from treasure import Treasure

def priority_comp_fn(x,y): #current time is the time of insertion of the object
    current_time=0#since it doesnt make a difference in comparator as it its used in both
    p1=(current_time-x.arrival_time-x.size,x.id)
    p2=(current_time-y.arrival_time-y.size,y.id)
    return p1>p2 #since we want maxheap based on priority

def crew_comp_fn(x,y):
    return x.working_time<y.working_time

class StrawHatTreasury:
    '''
    Class to implement the StrawHat Crew Treasury
    '''
    
    def __init__(self, m):
        '''
        Arguments:
            m : int : Number of Crew Mates (positive integer)
        Returns:
            None
        Description:
            Initializes the StrawHat
        Time Complexity:
            O(m)
        '''
        self.current_time=0
        self.treasure_arr=[]
        self.crew_arr=[]
        self.crew_heap=Heap(crew_comp_fn,[]) #initialize a heap of crew members
        # Write your code here
        for i in range(m):
            crewmem=CrewMate(i)
            self.crew_heap.insert(crewmem) #each element of the crew heap stores an instance of crewmate class
            self.crew_arr.append(crewmem)
        pass
    
    def add_treasure(self, treasure): #treasure - id, size, arrival time
        '''
        Arguments:
            treasure : Treasure : The treasure to be added to the treasury
        Returns:
            None
        Description:
            Adds the treasure to the treasury
        Time Complexity:
            O(log(m) + log(n)) where
                m : Number of Crew Mates
                n : Number of Treasures
        '''
        
        # Write your code here
        time_now=treasure.arrival_time
        current_member=self.crew_heap.extract()#take out the member with least load
        current_member.add_treasure(treasure) #adds treasure to this crew member
        self.crew_heap.insert(current_member)#insert back into the heap !! change this function in heap.py

        self.treasure_arr.append(treasure)#add to treasure array
        pass
    
    def get_completion_time(self):
        '''
        Arguments:
            None
        Returns:
            List[Treasure] : List of treasures in the order of their ids after updating Treasure.completion_time
        Description:
            Returns all the treasure after processing them
        Time Complexity:
            O(n(log(m) + log(n))) where
                m : Number of Crew Mates
                n : Number of Treasures
        '''
        # Write your code here
        self.completion_arr=[]
        for ele in self.crew_arr:
            lis=ele.get_completion_time_for_member()
            self.completion_arr.extend(lis)
        sorted_li=sorted(self.completion_arr,key=lambda x: x.id)
        return sorted_li
    
    # You can add more methods if required


if __name__=="__main__":
    treasury=StrawHatTreasury(3)
    t1=Treasure(1,8,1)
    t2=Treasure(2,7,2)
    t3=Treasure(3,4,4)
    t4=Treasure(4,1,5)
    treasury.add_treasure(t1)
    treasury.add_treasure(t2)
    treasury.add_treasure(t3)
    treasury.add_treasure(t4)
    for i in treasury.crew_arr:
        print(i.treasure_info_arr)
    
    '''treasury=StrawHatTreasury(2)
    t1=Treasure(1000, 1000000000, 1)
    t2=Treasure(1001, 2000000000, 300000000)
    t3=Treasure(1002, 100000000, 400000000)
    t4=Treasure(1003, 5000000000, 600000000)
    t5=Treasure(1004, 1200000000, 700000000)
    treasury.add_treasure(t1)
    treasury.add_treasure(t2)
    treasury.add_treasure(t3)
    treasury.add_treasure(t4)
    treasury.add_treasure(t5)
    for i in treasury.crew_arr:
        print(i.treasure_info_arr)'''

'''
Add 1000 1000000000 1
Add 1001 2000000000 300000000
Add 1002 100000000 400000000
Add 1003 5000000000 600000000
Add 1004 1200000000 700000000'''