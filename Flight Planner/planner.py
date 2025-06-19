from flight import Flight

class Planner:
    def __init__(self, flights):
        """The Planner

        Args:
            flights (List[Flight]): A list of information of all the flights (objects of class Flight)
        """
        self.graph = []
        max_city = 0
        for flight in flights:
            max_city = max(max_city, flight.start_city, flight.end_city)
        self.graph = [[] for _ in range(max_city + 1)] # Initialize graph with empty lists for each city
        for flight in flights:
            self.graph[flight.start_city].append(flight)
    
    def least_flights_earliest_route(self, start_city, end_city, t1, t2):
        """
        Return List[Flight]: A route from start_city to end_city, which departs after t1 (>= t1) and
        arrives before t2 (<=) satisfying: 
        The route has the least number of flights, and within routes with same number of flights, 
        arrives the earliest
        """
        num_cities = len(self.graph)
        queue = Queue()  
        queue.enqueue((start_city, [], t1-20)) 
        min_flights = [float('inf')] * num_cities #minimum flights to reach each city
        arrival_time = [float('inf')] * num_cities #earliest arrival time to reach each city
        
        min_flights[start_city] = 0
        arrival_time[start_city] = t1
        best_route = None
        best_arrival_time = float('inf')
        
        while not queue.is_empty():
            current_city, path, current_time = queue.dequeue()
            if current_city == end_city:
                if best_route==None and current_time<best_arrival_time: #if no route found yet
                    best_route = path
                    best_arrival_time = current_time
                elif len(path) < len(best_route): #if found a route with fewer flights, update best_route and best_arrival_time
                    best_route = path
                    best_arrival_time = current_time
                elif len(path)==len(best_route) and current_time<best_arrival_time: 
                    #if found a route with same number of flights but earlier arrival time, update best_route and best_arrival_time
                    best_route = path
                    best_arrival_time = current_time
                continue # dont explore further from this city; go to the next city in the queue
            for flight in self.graph[current_city]: # Explore all flights from the current city
                # Check if the flight meets the general time constraints
                if flight.departure_time >= t1 and \
                   flight.departure_time >= current_time + 20 and \
                   flight.arrival_time <= t2:
                    
                    new_time = flight.arrival_time
                    flights_taken = len(path) + 1
                    
                    # Proceed only if this path is better (fewer flights or earlier arrival time for the same number of flights)
                    if flights_taken < min_flights[flight.end_city] or \
                       (flights_taken == min_flights[flight.end_city] and new_time < arrival_time[flight.end_city]):
                        min_flights[flight.end_city] = flights_taken
                        arrival_time[flight.end_city] = new_time
                        #add to the queue the edn city with the new path and arrival time
                        queue.enqueue((flight.end_city, path + [flight], new_time)) 
        if best_route==None:
            return []
        return best_route
    

    def cheapest_route(self, start_city, end_city, t1, t2):
        """
        Return List[Flight]: A route from start_city to end_city, which departs after t1 (>= t1) and
        arrives before t2 (<=) satisfying: 
        The route is a cheapest route
        """
        num_cities = len(self.graph)
        heap = Heap(lambda x, y: x[0] < y[0],[(0, start_city, [], t1-20)])
        min_fare = [float('inf')] * num_cities # minimum fare to reach each city
        min_fare[start_city] = 0
        best_route = None
        
        while len(heap.heap)!=0:
            cumulative_fare, current_city, path, current_time = heap.extract()
            
            #if the end city is reached, the path is the best route since the heap is a min-heap on fares
            #since it is a min-heap on fares, the element extracted is the cheapaest route to the current city.
            #hence, if the current city is the end city, we have found the best route
            if current_city == end_city:
                best_route = path
                break

            for flight in self.graph[current_city]:
                # Check if the flight meets the general time constraints
                if flight.departure_time >= t1 and \
                   flight.departure_time >= current_time + 20 and \
                   flight.arrival_time <= t2:
                    new_fare = cumulative_fare + flight.fare
                    new_time = flight.arrival_time
                    # Proceed only if this path is cheaper
                    if new_fare < min_fare[flight.end_city]:
                        min_fare[flight.end_city] = new_fare
                        heap.insert((new_fare, flight.end_city, path + [flight], new_time))
        if best_route==None:
            return []
        return best_route
    
    def least_flights_cheapest_route(self, start_city, end_city, t1, t2):
        """
        Return List[Flight]: A route from start_city to end_city, which departs after t1 (>= t1) and
        arrives before t2 (<=) satisfying: 
        The route has the least number of flights, and within routes with same number of flights, 
        is the cheapest
        """
        #the logic is similar to the least_flights_earliest_route.
        #but instead of tracking arrival time, we track fare for adding the path to the queue.
        num_cities = len(self.graph)
        queue = Queue()
        queue.enqueue((0, 0, start_city, [], t1-20))
        min_flights = [float('inf')] * num_cities
        min_fare = [float('inf')] * num_cities
        min_flights[start_city] = 0
        min_fare[start_city] = 0
        best_route = None
        
        while not queue.is_empty():
            flights_count, cumulative_fare, current_city, path, current_time = queue.dequeue()
            if current_city == end_city:
                if best_route==None :
                        best_route = path
                        min_fare[end_city] = cumulative_fare
                elif len(path) < len(best_route):
                    best_route = path
                    min_fare[end_city] = cumulative_fare
                elif len(path)==len(best_route) and cumulative_fare<min_fare[end_city]:
                    best_route = path
                    min_fare[end_city] = cumulative_fare
                continue

            # Explore all flights from the current city
            for flight in self.graph[current_city]:
                # Check if the flight meets time constraints
                if flight.departure_time >= t1 and \
                   flight.departure_time >= current_time + 20 and \
                   flight.arrival_time <= t2:
                    new_time = flight.arrival_time
                    new_fare = cumulative_fare + flight.fare
                    new_flights_count = flights_count + 1

                    # Proceed only if this path is better (fewer flights or cheaper fare for the same number of flights)
                    if new_flights_count < min_flights[flight.end_city] or \
                       (new_flights_count == min_flights[flight.end_city] and new_fare < min_fare[flight.end_city]):
                        min_flights[flight.end_city] = new_flights_count
                        min_fare[flight.end_city] = new_fare
                        queue.enqueue((new_flights_count, new_fare, flight.end_city, path + [flight], new_time))

        if best_route is not None: return best_route 
        else: return []

class Node:
    #A node in the linked list used for the queue
    def __init__(self, value):
        self.value = value
        self.next = None

class Queue:
    #A queue implemented using a linked list, supporting O(1) enqueue and dequeue operations.
    def __init__(self):
        self.head = None  # Points to the front of the queue
        self.tail = None  # Points to the end of the queue
        self.size = 0     # Track the number of items in the queue

    def enqueue(self, item): # Add an item to the end of the queue
        new_node = Node(item)
        if self.tail:  # Queue is not empty
            self.tail.next = new_node
        else:  # Queue is empty, so head should also point to the new node
            self.head = new_node
        self.tail = new_node
        self.size += 1

    def dequeue(self): # Remove an item from the front of the queue
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        value = self.head.value
        self.head = self.head.next  # Move head to the next node
        if not self.head:  # Queue is now empty, so tail should also be None
            self.tail = None
        self.size -= 1
        return value

    def is_empty(self):
        return self.size == 0

    def front(self): # Return the item at the front of the queue without removing it
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self.head.value

    def get_size(self):
        return self.size
    

'''
Python Code to implement a heap with general comparison function
'''

class Heap:
    '''
    Class to implement a min-heap with general comparison function
    '''
    
    def __init__(self, comparison_function, init_array=[]):
        self.compare = comparison_function
        self.heap = []
        for value in init_array:
            self.insert(value)

    def insert(self, value):
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def extract(self):
        if len(self.heap) == 0:
            raise IndexError("extract from an empty heap")
        
        root_value = self.heap[0]
        # Move the last element to the root and remove the last element
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        # Restore the heap property by heapifying down
        if len(self.heap) > 0:
            self._heapify_down(0)
        
        return root_value
    
    def top(self):
        if len(self.heap) == 0:
            raise IndexError("top from an empty heap")
        
        return self.heap[0]
    
    def _heapify_up(self, index):
        parent = (index - 1) // 2
        if index > 0 and self.compare(self.heap[index], self.heap[parent]):
            # Swap if the current element is "smaller" than the parent
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self._heapify_up(parent)

    def _heapify_down(self, index):

        left_child = 2 * index + 1
        right_child = 2 * index + 2
        smallest = index
        
        # Check if left child is smaller
        if left_child < len(self.heap) and self.compare(self.heap[left_child], self.heap[smallest]):
            smallest = left_child
        
        # Check if right child is smaller
        if right_child < len(self.heap) and self.compare(self.heap[right_child], self.heap[smallest]):
            smallest = right_child
        
        # If the smallest element is not the current one, swap and continue
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._heapify_down(smallest)
