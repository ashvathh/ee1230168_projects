from prime_generator import get_next_size

def num(char):
    if 'a'<=char<='z':
       return ord(char)-97
    else:
        return ord(char)-39

class HashTable:
    def __init__(self, collision_type, params):
        '''
        Possible collision_type:
            "Chain"     : Use hashing with chaining
            "Linear"    : Use hashing with linear probing
            "Double"    : Use double hashing
        '''
        self.collision_type=collision_type
        if collision_type=="Chain":
            self.z,self.table_size=params
            self.table=[[] for i in range(self.table_size)]
        elif collision_type=="Linear":
            self.z,self.table_size=params
            self.table=[None]*self.table_size
        elif collision_type=="Double":
            self.z1,self.z2,self.c2,self.table_size=params
            self.table=[None]*self.table_size
        self.size_used=0

    def polynomial_hash(self,key,z,table_size):
        hash_value=0
        for i,char in enumerate(key):
            hash_value=(hash_value*z+ord(char))%table_size
        return hash_value
    
    def insert(self, x):
        if isinstance(x,tuple):key=x[0]
        else: key=x
        if self.collision_type=="Chain":
            hash_key=self.polynomial_hash(key,self.z,self.table_size)
            self.table[hash_key].append(x)
            self.size_used+=1
        elif self.collision_type=="Linear":
            hash_key=self.polynomial_hash(key,self.z,self.table_size)
            for i in range(self.table_size):
                slot=(hash_key+i)%self.table_size
                if self.table[slot] is None:
                    self.table[slot] = x
                    self.size_used += 1
                    break
        elif self.collision_type=="Double":
            h1 = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2-self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot = (h1 + i * h2) % self.table_size
                if self.table[slot] is None:
                    self.table[slot] = x
                    self.size_used += 1
                    break
        pass
    
    def find(self, key):
        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for kv in self.table[hash_key]:
                if isinstance(kv,tuple):
                    if kv[0]==key:
                        return kv[1]
                else:
                    if kv==key:
                        return True
                    x=1
                    key1=kv[0]
                    if key1 == key: 
                        return kv[1]
            return None if any(isinstance(item, tuple) for item in self.table[hash_key] if item is not None) else False 
        
        elif self.collision_type=="Linear":
            for i in range(self.table_size):
                hash_key = self.polynomial_hash(key, self.z, self.table_size)
                slot=(hash_key + i)%self.table_size
                if self.table[slot] is None:  
                    return False
                if isinstance(self.table[slot],tuple):
                    if self.table[slot][0]==key:
                        return self.table[slot][1]
                else:
                    if self.table[slot] == key:
                        return True
            return None if any(isinstance(item, tuple) for item in self.table if item is not None) else False 
        
        elif self.collision_type=="Double":
            h1 = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2-self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot=(h1 + i*h2)%self.table_size
                if self.table[slot] is None:  # if we reach to an empty slot, then the key is not present there.
                    return False
                if isinstance(self.table[slot],tuple):
                    if self.table[slot][0]==key:
                        return self.table[slot][1]
                else:
                    if self.table[slot] == key:
                        return True
            return None if any(isinstance(item, tuple) for item in self.table if item is not None) else False 

                
    
    def get_slot(self, key):

        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            return hash_key

        elif self.collision_type == "Linear":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:  # stop if an empty slot is reached after iterating
                    return slot
                if self.table[slot] == key:  # return the slot if the key is found at that slot
                    return slot

        elif self.collision_type == "Double":
            h1 = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2-self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot = (hash_key + i * (self.z2 or 1)) % self.table_size
                if self.table[slot] is None:  # stop
                    return slot
                if self.table[slot] == key:  # check n return
                    return slot

        return None  # if key is not in the table
    
    def get_load(self):
        return self.size_used/self.table_size
    
    def __str__(self):
        output = []
    
        for slot in self.table:
            if slot is None:
                output.append("<EMPTY>") # if the slot is empty
            elif isinstance(slot, list):
                if all(isinstance(item, tuple) for item in slot):
                    output.append(" ; ".join(f"({key}, {value})" for key, value in slot))
                else:  # HashSet
                    output.append(" ; ".join(f"{key}" for key in slot))
            else:
                if isinstance(slot, tuple):
                    output.append(f"({slot[0]}, {slot[1]})")
                else:
                    output.append(f"{slot}")
        
        return " | ".join(output) #this is the final output string that we output
    
    # TO BE USED IN PART 2 (DYNAMIC HASH TABLE)
    def rehash(self):
        pass
    
# IMPLEMENT ALL FUNCTIONS FOR CLASSES BELOW
# IF YOU HAVE IMPLEMENTED A FUNCTION IN HashTable ITSELF, 
# YOU WOULD NOT NEED TO WRITE IT TWICE
    
class HashSet(HashTable):
    def __init__(self, collision_type, params):
        self.collision_type=collision_type
        if collision_type=="Chain":
            self.z,self.table_size=params
            self.table=[[] for i in range(self.table_size)]
        elif collision_type=="Linear":
            self.z,self.table_size=params
            self.table=[None]*self.table_size
        elif collision_type=="Double":
            self.z1,self.z2,self.c2,self.table_size=params
            self.table=[None]*self.table_size
        self.size_used=0
    
    def num(char):
        if 'a'<=char<='z':
            return ord(char)-97
        else:
            return ord(char)-39

    def polynomial_hash(self,key,z,table_size):
        hash_value=0
        key=key[::-1]
        for char in key:
            hash_value=(hash_value*z+num(char))%table_size
        return hash_value
    

    def insert(self, key):

        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            if key not in self.table[hash_key]:
                self.table[hash_key].append(key)
                self.size_used+=1

        elif self.collision_type == "Linear":
            # linear probing
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:
                    self.table[slot] = key
                    self.size_used+=1
                    return
                elif self.table[slot] == key: 
                    return 

        elif self.collision_type == "Double":
            # double hashing
            hash_key = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2 - self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot = (hash_key + i * h2) % self.table_size
                if self.table[slot] is None: 
                    self.table[slot] = key
                    self.size_used+=1
                    return
                elif self.table[slot] == key: 
                    return 
    
    def find(self, key):
        #hash_key = self.polynomial_hash(key, self.z1, self.table_size)

        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            if self.table[hash_key] is not None:
                return key in self.table[hash_key]
            return False

        elif self.collision_type == "Linear":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:  # first empty slot is hit - no key
                    return False
                if self.table[slot] == key:  # key is hit
                    return True
            return False 

        elif self.collision_type == "Double":
            # double hashing
            hash_key = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2 - self.polynomial_hash(key, self.z2, self.c2) or 1
            for i in range(self.table_size):
                slot = (hash_key + i * h2) % self.table_size
                if self.table[slot] is None:  # first empty slot is hit - no key
                    return False
                if self.table[slot] == key:  # key is reached
                    return True
            return False 
    
    def get_slot(self, key):
        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            return hash_key 

        elif self.collision_type == "Linear":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:
                    return slot
                if self.table[slot] == key:
                    return slot

        elif self.collision_type == "Double":
            h1 = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2-self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot = (h1 + i * (h2)) % self.table_size
                if self.table[slot] is None:
                    return slot
                if self.table[slot] == key:
                    return slot

        return None
    
    def get_load(self):
        return self.size_used/self.table_size
    
    def __str__(self):
        output = []
    
        for slot in self.table:
            if slot is None or slot==[]:
                output.append("<EMPTY>")
            elif isinstance(slot, list):
                if all(isinstance(item, tuple) for item in slot):
                    output.append(" ; ".join(f"({key}, {value})" for key, value in slot))
                else:
                    output.append(" ; ".join(f"{key}" for key in slot))
            else:
                if isinstance(slot, tuple):
                    output.append(f"({slot[0]}, {slot[1]})")
                else:
                    output.append(f"{slot}")
        
        return " | ".join(output) #this is the final output string that we output
    
class HashMap(HashTable): #we use x[0] in the comparator. methods remain same as hashset
    def __init__(self, collision_type, params):
        self.collision_type=collision_type
        if collision_type=="Chain":
            self.z,self.table_size=params
            self.table=[[] for i in range(self.table_size)]
        elif collision_type=="Linear":
            self.z,self.table_size=params
            self.table=[None]*self.table_size
        elif collision_type=="Double":
            self.z1,self.z2,self.c2,self.table_size=params
            self.table=[None]*self.table_size
        self.size_used=0
        pass
    
    def polynomial_hash(self,key,z,table_size):
        hash_value=0
        key=key[::-1]
        for char in key:
            hash_value=(hash_value*z+num(char))%table_size
        return hash_value

    def insert(self, x):
        # x = (key, value)
        key=x[0]
        #print(x)
        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            # only insert if the key is not already present
            if x not in self.table[hash_key]:
                self.table[hash_key].append(x)
                self.size_used+=1

        elif self.collision_type == "Linear":
            # linear probing
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:
                    self.table[slot] = x
                    self.size_used+=1
                    return
                elif self.table[slot] == x:
                    
                    return 

        elif self.collision_type == "Double":
            hash_key = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2 - self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot = (hash_key + i * h2) % self.table_size
                if self.table[slot] is None:
                    self.table[slot] = x
                    self.size_used+=1
                    return
                elif self.table[slot] == x:
                    return 

        pass
    
    def find(self, key):
        #hash_key = self.polynomial_hash(key, self.z1, self.table_size)

        if self.collision_type == "Chain":
            #print("in chain")
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            if self.table[hash_key]!=[]:
                for ele in self.table[hash_key]:
                    if ele[0]==key:
                        return ele[1]
                #return key in self.table[hash_key]
            return None

        elif self.collision_type == "Linear":
            #print("in linear")
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:
                    return None
                if self.table[slot][0] == key:
                    return self.table[slot][1]
            return None 

        elif self.collision_type == "Double":
            #print("in double")
            hash_key = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2 - self.polynomial_hash(key, self.z2, self.c2) or 1
            for i in range(self.table_size):
                slot = (hash_key + i * h2) % self.table_size
                if self.table[slot] is None:
                    return None
                if self.table[slot][0] == key:
                    return self.table[slot][1]
            print("key not found")
            return None
        pass
    
    def get_slot(self, key):
        if self.collision_type == "Chain":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            return hash_key

        elif self.collision_type == "Linear":
            hash_key = self.polynomial_hash(key, self.z, self.table_size)
            for i in range(self.table_size):
                slot = (hash_key + i) % self.table_size
                if self.table[slot] is None:
                    return slot
                if self.table[slot][0] == key:
                    return slot

        elif self.collision_type == "Double":
            h1 = self.polynomial_hash(key, self.z1, self.table_size)
            h2 = self.c2-self.polynomial_hash(key, self.z2, self.c2)
            for i in range(self.table_size):
                slot = (h1 + i * (h2)) % self.table_size
                if self.table[slot] is None:
                    return slot
                if self.table[slot][0] == key:
                    return slot

        return None
    
    def get_load(self):
        return self.size_used/self.table_size
    
    def __str__(self):
        output=[]
        for slot in self.table:
            if self.collision_type=="Chain":
                if slot:
                    output.append(';'.join(map(str,slot)))
                else:
                    output.append("<EMPTY>")
            else:
                if slot is not None:
                    output.append(str(slot))
                else:
                    output.append("<EMPTY>")
            return " | ".join(output)
