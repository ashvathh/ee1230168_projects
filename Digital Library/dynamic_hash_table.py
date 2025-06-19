from hash_table import HashSet, HashMap
from prime_generator import get_next_size

class DynamicHashSet(HashSet):
    def __init__(self, collision_type, params):
        super().__init__(collision_type, params)
        
    def rehash(self):
        # IMPLEMENT THIS FUNCTION
        self.table_size=get_next_size() #new size of the table
        old_table=self.table #creat a copy of the old table
        self.size_used=0
        if self.collision_type=="Chain":
            self.table=[[] for i in range(self.table_size)]
        else:
            self.table=[None]*self.table_size
        
        if self.collision_type=="Chain":
            for slot in old_table:
                for ele in slot:
                    new_slot=self.get_slot(ele)
                    self.table[new_slot].append(ele)
                    self.size_used+=1

        else:#insert at first empty cell
            if self.collision_type=="Linear":
                for ele in old_table:
                    hash_key=self.polynomial_hash(ele,self.z,self.table_size)
                    for i in range(self.table_size):
                        slot=(hash_key+i)%self.table_size
                        if self.table[slot] is None:
                            self.table[slot] = ele
                            self.size_used += 1
                            break

            if self.collision_type=="Double":
                for ele in old_table:
                    h1=self.polynomial_hash(ele,self.z1,self.table_size)
                    h2=self.c2-self.polynomial_hash(ele,self.z2,self.c2)
                    for i in range(self.table_size):
                        slot=(h1+i*h2)%self.table_size
                        if self.table[slot] is None:  # check this only when the slot is empty
                            self.table[slot] = ele
                            self.size_used += 1
                            break
        
    def insert(self, x):
        # YOU DO NOT NEED TO MODIFY THIS
        super().insert(x)
        
        if self.get_load() >= 0.5:
            self.rehash()
            
            
class DynamicHashMap(HashMap):
    def __init__(self, collision_type, params):
        super().__init__(collision_type, params)
        
    def rehash(self):
        # IMPLEMENT THIS FUNCTION
        self.table_size=get_next_size() #new size of the table
        old_table=self.table #creat a copy of the old table
        self.size_used=0
        if self.collision_type=="Chain":
            self.table=[[] for i in range(self.table_size)]
        else:
            self.table=[None]*self.table_size
        
        if self.collision_type=="Chain":
            for slot in old_table:
                for ele in slot:
                    new_slot=self.get_slot(ele[0])
                    self.table[new_slot].append(ele)
                    self.size_used+=1

        else:#insert at first empty cell
            if self.collision_type=="Linear":
                for ele in old_table:
                    hash_key=self.polynomial_hash(ele[0],self.z,self.table_size)
                    for i in range(self.table_size):
                        slot=(hash_key+i)%self.table_size
                        if self.table[slot] is None:  
                            self.table[slot] = ele
                            self.size_used += 1
                            break

            if self.collision_type=="Double":
                for ele in old_table:
                    h1=self.polynomial_hash(ele[0],self.z1,self.table_size)
                    h2=self.c2-self.polynomial_hash(ele[0],self.z2,self.c2)
                    for i in range(self.table_size):
                        slot=(h1+i*h2)%self.table_size
                        if self.table[slot] is None:  
                            self.table[slot] = ele
                            self.size_used += 1
                            break
        
        pass
        
    def insert(self, key):
        # YOU DO NOT NEED TO MODIFY THIS
        super().insert(key)
        
        if self.get_load() >= 0.5:
            self.rehash()
