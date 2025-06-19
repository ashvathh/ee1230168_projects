class Flight:
    def __init__(self, flight_no, start_city, departure_time, end_city, arrival_time, fare):
        self.flight_no = flight_no # Unique ID of each flight
        self.start_city = start_city # The city no. where the flight starts
        self.departure_time = departure_time # Time at which the flight starts
        self.end_city = end_city # The city no. where the flight ends
        self.arrival_time = arrival_time # Time at which the flight ends
        self.fare = fare # The cost of taking this flight
        
"""
If there are n flights, and m cities:

1. Flight No. will be an integer in {0, 1, ... n-1}
2. Cities will be denoted by an integer in {0, 1, .... m-1}
3. Time is denoted by a non negative integer - we model time as going from t=0 to t=inf
"""
