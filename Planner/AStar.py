# kernprof -l -v AStar.py
# does not select min(g+h) from open list
'''
autoware.ai/src/autoware/core_planning/astar_search/src/astar_search.cpp
pythonrobotics
'''
import numpy as np
from queue import Queue
import sys
import math
MAP_WIDTH = 8
MAP_HEIGHT = 8
class Point:
    def __init__(self, row, col, parent):
        self.row_ = row
        self.col_ = col
        self.parent_ = parent
    def __str__(self):
        if self.parent_ != None:
            return '%2d%2d%2d%2d' % (self.row_, self.col_, self.parent_.row_, self.parent_.col_) 
        else:
            return '%2d%2d%2d%2d' % (self.row_, self.col_, 0, 0) 
def generate_map(width, height):
    origin_map = np.zeros((height, width))
    start_point = Point(0, 0, None)
    goal_point = Point(0, height//2, None)
    origin_map[:, 2] = np.ones(height)
    origin_map[height-1, 2] = 0
    origin_map[3*height//4, 4:] = np.ones((1, width-4))
    origin_map[2*height//4, 3:-2] = np.ones((1, width-5))
    return start_point, goal_point, origin_map



def AStar(start_point, goal_point, origin_map):
    height, width = origin_map.shape
    origin_map[start_point.row_, start_point.col_] = 3
    origin_map[goal_point.row_, goal_point.col_] = 4

    def distance(p, p_):
        return math.sqrt((p.row_ - p_.row_)**2 + (p.col_ - p_.col_)**2)
    # def g(cur_point):
    #     return cur_point.row_ + cur_point.col_ - start_point.row_ - start_point.col_
    def h(cur_point):
        return math.sqrt((cur_point.row_ - goal_point.row_)**2 + (cur_point.col_ - goal_point.col_)**2)
        # return abs(cur_point.row_ - goal_point.row_) + abs(cur_point.col_ - goal_point.col_)

    def nghbr_vis(cur_point):
        nghbr_list = []

        x_offset = y_offset = np.array([-1, 0, 1])
        x_offsets, y_offsets = np.meshgrid(x_offset, y_offset)
        for x,y in zip(x_offsets.flatten(), y_offsets.flatten()):
            if x == 0 and y == 0:
                continue
            if cur_point.row_+y >= 0 and cur_point.col_+x >= 0 \
                and cur_point.row_+y < height and cur_point.col_+x < width \
                and origin_map[cur_point.row_+y , cur_point.col_+x]!=1 :
                nghbr_list.append(Point(cur_point.row_+y,cur_point.col_+x, cur_point))
        
        return nghbr_list

    close_map = np.zeros((height, width))
    open_map = np.zeros((height, width))
    gfunc_map = np.ones((height, width))*100
    open_list = []
    @profile
    def compute_cost(p, p_):
        if gfunc_map[p.row_, p.col_] + distance(p, p_) < gfunc_map[p_.row_, p_.col_]:
            if p_.parent_  != None:
                print(p_.parent)
            p_.parent_ = p
            gfunc_map[p_.row_, p_.col_] = gfunc_map[p.row_, p.col_] + distance(p, p_)
    @profile
    def update_vertex(p, p_):
        g_old = gfunc_map[p_.row_, p_.col_]
        compute_cost(p, p_)
        if gfunc_map[p_.row_, p_.col_] < g_old:
            if open_map[p_.row_, p_.col_] == 1:
                open_map[p_.row_, p_.col_] = 0
            # if p_ in open_list:
                open_list.remove(p_)
                print("remove:", p_)
                # [open_list.remove(p_) for p in open_list if p.row_==p_.row_ and p.col_==p_.col_]
            open_list.append((p_, gfunc_map[p_.row_, p_.col_] + h(p_)))
            # open_map[p_.row_, p_.col_] == 1

    start_point.parent = start_point

    gfunc_map[start_point.row_][start_point.col_] = 0
    open_list.append((start_point, gfunc_map[start_point.row_][start_point.col_]+ h(start_point)))

    while (len(open_list) > 0):
        point, cost_value = open_list.pop(0)
        if point.row_ == goal_point.row_ and point.col_ == goal_point.col_:
            return point
        
        close_map[point.row_, point.col_] = 1
        for p in nghbr_vis(point):

            if close_map[p.row_, p.col_] != 1:
                if open_map[p.row_, p.col_] != 1:
                # if p not in open_list:
                    gfunc_map[p.row_][p.col_] = sys.maxsize
                    p.parent_ = None
                update_vertex(point, p)


    return Point(sys.maxsize, sys.maxsize, None)

start_point, goal_point, origin_map = generate_map(MAP_WIDTH, MAP_HEIGHT)

print(start_point)
print(origin_map)
end_point = AStar(start_point, goal_point, origin_map)
print(end_point)
print(end_point.parent_)

if end_point.parent_ == None:
    print("Not Found")
    exit(0)

end_point = end_point.parent_
while end_point.row_ != start_point.row_ or end_point.col_ != start_point.col_:
    origin_map[end_point.row_][end_point.col_] = 2
    end_point = end_point.parent_
    print("end_point:", end_point)

print(origin_map)
