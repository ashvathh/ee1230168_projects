import hash_table as ht

class DigitalLibrary:
    @staticmethod
    def mergesort_text(arr):
        if len(arr)==1:
            return arr
        mid=len(arr)//2
        l_arr=arr[:mid]
        r_arr=arr[mid:]

        l_arr=DigitalLibrary.mergesort_text(l_arr)
        r_arr=DigitalLibrary.mergesort_text(r_arr)

        merged_arr=[]
        i=j=0
        while i<len(l_arr) and j<len(r_arr):
            if l_arr[i]<r_arr[j]:
                merged_arr.append(l_arr[i])
                i+=1
            elif l_arr[i]==r_arr[j]:
                merged_arr.append(l_arr[i])
                i+=1
                j+=1
            else:
                merged_arr.append(r_arr[j])
                j+=1
        while i < len(l_arr):
            merged_arr.append(l_arr[i])
            i += 1
        while j < len(r_arr):
            merged_arr.append(r_arr[j])
            j += 1
        return merged_arr

    @staticmethod
    def mergesort_book(arr):
        if len(arr)==1:
            return arr
        mid=len(arr)//2
        l_arr=arr[:mid]
        r_arr=arr[mid:]

        l_arr=DigitalLibrary.mergesort_book(l_arr)
        r_arr=DigitalLibrary.mergesort_book(r_arr)

        merged_arr=[]
        i=j=0
        while i<len(l_arr) and j<len(r_arr):
            if l_arr[i][0]<r_arr[j][0]:
                merged_arr.append(l_arr[i])
                i+=1
            elif l_arr[i][0]==r_arr[j][0]:
                merged_arr.append(l_arr[i])
                i+=1
                j+=1
            else:
                merged_arr.append(r_arr[j])
                j+=1
        while i < len(l_arr):
            merged_arr.append(l_arr[i])
            i += 1
        while j < len(r_arr):
            merged_arr.append(r_arr[j])
            j += 1
        return merged_arr

    @staticmethod
    def binarysearch(title, arr, l=0, h=None):
        if h is None:
            h = len(arr) - 1
        if l > h:
            return -1  # base case
        mid = (l + h) // 2
        if title < arr[mid][0]:
            return DigitalLibrary.binarysearch(title, arr, l, mid - 1)
        elif title > arr[mid][0]:
            return DigitalLibrary.binarysearch(title, arr, mid + 1, h)
        else:
            return mid
    
    @staticmethod
    def binarysearch1(title, arr, l=0, h=None):
        if h is None:
            h = len(arr) - 1
        if l > h:
            return -1  # base case
        mid = (l + h) // 2
        if title < arr[mid]:
            return DigitalLibrary.binarysearch1(title, arr, l, mid - 1)
        elif title > arr[mid]:
            return DigitalLibrary.binarysearch1(title, arr, mid + 1, h)
        else:
            return mid


        # DO NOT CHANGE FUNCTIONS IN THIS BASE CLASS
    def __init__(self):
        pass
    
    def distinct_words(self, book_title):
        pass
    
    def count_distinct_words(self, book_title):
        pass
    
    def search_keyword(self, keyword):
        pass
        
    def print_books(self):
        pass
    
class MuskLibrary(DigitalLibrary):
    # IMPLEMENT ALL FUNCTIONS HERE
    def __init__(self, book_titles, texts):
        self.book_arr=[]
        for i in range(len(book_titles)):
            sorted_text=DigitalLibrary.mergesort_text(texts[i])
            self.book_arr.append([book_titles[i],sorted_text])
        self.book_arr=DigitalLibrary.mergesort_book(self.book_arr)
    
    def distinct_words(self, book_title):
        book_ind=DigitalLibrary.binarysearch(book_title,self.book_arr)
        if book_ind!=-1:
            return self.book_arr[book_ind][1]
        else: return []
    
    def count_distinct_words(self, book_title):
        book_ind=DigitalLibrary.binarysearch(book_title,self.book_arr)
        if book_ind!=-1:
            return len(self.book_arr[book_ind][1])
        else: return 0
    
    def search_keyword(self, keyword):
        keyword_arr=[]
        for book in self.book_arr:
            ind=DigitalLibrary.binarysearch1(keyword,book[1])
            if ind!=-1:
                keyword_arr.append(book[0])
        return keyword_arr
    
    def print_books(self):
        str=''
        for book in self.book_arr:
            str+=book[0]+": "
            for word in book[1]:
                str+=word+" | "
            str=str[:-3]+'\n'
        print(str)


class JGBLibrary(DigitalLibrary):
    # IMPLEMENT ALL FUNCTIONS HERE
    def __init__(self, name, params):
        '''
        name    : "Jobs", "Gates" or "Bezos"
        params  : Parameters needed for the Hash Table:
            z is the parameter for polynomial accumulation hash
            Use (mod table_size) for compression function
            
            Jobs    -> (z, initial_table_size)
            Gates   -> (z, initial_table_size)
            Bezos   -> (z1, z2, c2, initial_table_size)
                z1 for first hash function
                z2 for second hash function (step size)
                Compression function for second hash: mod c2
        '''
        self.params=params
        if name=="Jobs": self.ct="Chain"
        if name=="Gates": self.ct="Linear"
        if name=="Bezos": self.ct="Double"
        self.hash_table=ht.HashMap(self.ct,self.params)
        
    
    def add_book(self, book_title, text):
        text_hashset=ht.HashSet(self.ct,self.params)
        for word in text:
            text_hashset.insert(word)
        self.hash_table.insert((book_title,text_hashset))
        #make a hashset with the words in the book
        pass
    
    def distinct_words(self, book_title):
        text=self.hash_table.find(book_title)
        ans_li=[]
        if self.ct=="Chain":
            for ele in text.table:
                if ele!=[]:
                    ans_li.extend(ele)
        else:
            for ele in text.table:
                if ele is not None:
                    ans_li.append(ele)
        return ans_li

        pass
    
    def count_distinct_words(self, book_title):
        text=self.hash_table.find(book_title)
        return text.size_used
    
    def search_keyword(self, keyword):
        ''''''
        book_name_arr=[]
        if self.ct=="Linear" or self.ct=="Double":
            for ele in self.hash_table.table:
                if ele is not None:
                    ind=ele[1].find(keyword)
                    #ind=DigitalLibrary.binarysearch1(keyword,ele[1].table)
                    if ele[1].find(keyword):
                        book_name_arr.append(ele[0])
        else:
            for slot in self.hash_table.table:
                for ele in slot:
                    #ind=DigitalLibrary.binarysearch1(keyword,ele[1].table)
                    if ele[1].find(keyword):
                        book_name_arr.append(ele[0])

        return book_name_arr
        ''''''
        pass
    
    def print_books(self):
        if self.ct=="Chain":
            for slot in self.hash_table.table:
                if slot!=[]:
                    for ele in slot:
                        print(f"{ele[0]}: ",end="")
                        print(f"{ele[1]}")
        else:
            for char in self.hash_table.table:
                if char:
                    print(f"{char[0]}: ",end="")
                    print(f"{char[1]}")
