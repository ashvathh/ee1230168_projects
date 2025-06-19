from navigator import *
from stack import *
from maze import *

def is_valid(x : int, y : int, rows : int, cols : int) -> bool:
    if(x < 0 or x > rows):
        print("Error in printing path, the row of cell ", (x, y), " is out of bounds")
        return False
    elif(y < 0 or y > cols):
        print("Error in printing path, the columns of cell ", (x, y), " is out of bounds")
        return False
    return True
def is_neighbour(x1 : int, y1 : int, x2 : int, y2 : int) -> bool:
    return abs(x2-x1) + abs(y2-y1) == 1
if __name__ == "__main__":
    ## YOU CAN TWEAK THESE PARAMETERS IN ORDER TO GENERATE MORE TESTCASES
    grid_rows = 4
    grid_cols = 4
    ghosts = [(1,1),(2,1),(2,2),(0,3),(3,0)]
    start_point = (0,0)
    end_point = (3,3)

    ## This is where the checker logic starts
    sample_grid = Maze(grid_rows, grid_cols)
    for ghost in ghosts:
        sample_grid.add_ghost(ghost[0], ghost[1])
    sample_grid.print_grid()
    try:
        PacManInstance = PacMan(sample_grid) 

        path = PacManInstance.find_path(start_point, end_point) 
        print(path)
        #print(path)
        for cell in path:
            if path not in ghosts:
                sample_grid.remove_ghost(cell[0],cell[1])
        #for i in range(len(path)-1):
        #    if path[i]==path[i+1]: path.pop(i+1)
        isPathValid = True
        if(path[0] != start_point):
            print("The path is supposed to begin with the tuple" , start_point)
            isPathValid = False
        if(path[-1] != end_point):
            print("The path is supposed to end with the tuple" , end_point)
            isPathValid = False
        for cell in path:
            if(not is_valid(cell[0], cell[1], grid_rows, grid_cols)):
                isPathValid = False
            if(sample_grid.grid_representation[cell[0]][cell[1]] == 1):
                print("The cell", cell, "that you have in your path is not vacant, hence this path is invalid.")
                isPathValid = False
        for i in range(len(path) - 1):
            if(not is_neighbour(path[i][0], path[i][1], path[i+1][0], path[i+1][1])):
                print("Cells ", path[i], "and", path[i+1], "are not neighbours. Your path is invalid!")
                isPathValid = False
        if(isPathValid):
            print("PATH FOUND SUCCESSFULLY!")    
        else:
            print("TESTCASE FAILED")    


    except (PathNotFoundException):
        print("TESTCASE FAILED. A VALID PATH DOES EXIST BETWEEN THESE TWO LOCATIONS")
    except:
        print("TESTCASE FAILED. YOU ARE NOT SUPPOSED TO RAISE ANY OTHER EXCEPTION APART FROM PathNotFoundException") 
