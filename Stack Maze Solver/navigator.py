from maze import *
from exception import *
from stack import *
    
def valid_move(x,y,maze_grid) -> bool:
    n_row=len(maze_grid)
    n_col=len(maze_grid[0])
    return ((x>=0 and x<n_row) and (y>=0 and y<n_col) and maze_grid[x][y]==0)

class PacMan:
    def __init__(self, grid : Maze) -> None:
        self.navigator_maze = grid.grid_representation
    def find_path(self, start : tuple, end : tuple) -> list:
        # IMPLEMENT FUNCTION HERE
        if self.navigator_maze[start[0]][start[1]]==1 or self.navigator_maze[end[0]][end[1]]==1:
            raise PathNotFoundException
                
        n_rows=len(self.navigator_maze)
        n_cols=len(self.navigator_maze[0])
        steps=[(1,0),(0,1),(-1,0),(0,-1)]

        path_stack=ArrayStack()
        path_stack.push(start)
        self.navigator_maze[start[0]][start[1]]=1
        while not path_stack.is_empty():
            current_cell=path_stack.top()
            if current_cell==end:
                #path_stack.push(stack.top())
                return path_stack.data
            
            for step in steps:
                new_cell=[0,0]
                new_cell[0]=current_cell[0]+step[0]
                new_cell[1]=current_cell[1]+step[1]
                new_cell=tuple(new_cell)
                if valid_move(new_cell[0],new_cell[1],self.navigator_maze):
                    path_stack.push(new_cell)
                    self.navigator_maze[new_cell[0]][new_cell[1]]=1
                    break

            #if stack.is_empty() or stack.top()!=current_cell:
            #    path_stack.pop()
            new_current_cell=path_stack.top()
            if new_current_cell==current_cell:
                path_stack.pop()
        #print(path_stack.data,"path data")
        raise PathNotFoundException
