from stack import *

class Maze:
    def __init__(self, m: int, n : int) -> None:
        ## DO NOT MODIFY THIS
        ## We initialise the list with all 0s, as initially all cells are vacant
        self.grid_representation = []
        for row in range(m):
            grid_row = []
            for column in range(n):
                grid_row.append(0)
            self.grid_representation.append(grid_row)
    
    def add_ghost(self, x : int, y: int) -> None:
        # IMPLEMENT YOUR FUNCTION HERE
        self.grid_representation[x][y]=1

    def remove_ghost(self, x : int, y: int) -> None:
        # IMPLEMENT YOUR FUNCTION HERE
        self.grid_representation[x][y]=0

    def is_ghost(self, x : int, y: int) -> bool:
        # IMPLEMENT YOUR FUNCTION HERE
        if self.grid_representation[x][y]==1: return True
        return False

    def print_grid(self) -> None:
        # IMPLEMENT YOUR FUNCTION HERE
        for i in range(len(self.grid_representation)):
            for j in range(len(self.grid_representation[0])):
                print(self.grid_representation[i][j],end=' ')
            print()
        return False
