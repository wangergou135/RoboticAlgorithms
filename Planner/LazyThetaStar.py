# kernprof -l -v LazyThetaStar.py
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
OBSTACLE_VALUE = 1
class Point:
    def __init__(self, row, col, parent):
        self.row_ = row
        self.col_ = col
        self.parent_ = parent
    def __str__(self):
        if self.parent_ != None:
            return '%2d%2d%2d%2d' % (self.row_, self.col_, self.parent_.row_, self.parent_.col_) 
        else:
            return '%2d%2d%2d%2d' % (self.row_, self.col_, -1, -1) 
def generate_map(width, height):
    origin_map = np.zeros((height, width))
    start_point = Point(0, 0, None)
    goal_point = Point(0, height//2, None)
    origin_map[:, 2] = np.ones(height)
    origin_map[height-1, 2] = 0
    origin_map[3*height//4, 4:] = np.ones((1, width-4))
    origin_map[2*height//4, 3:-2] = np.ones((1, width-5))
    return start_point, goal_point, origin_map



def LazyThetaStar(start_point, goal_point, origin_map):
    height, width = origin_map.shape
    origin_map[start_point.row_, start_point.col_] = 3
    origin_map[goal_point.row_, goal_point.col_] = 4

    close_map = np.zeros((height, width))
    open_map = np.zeros((height, width))
    gfunc_map = np.ones((height, width))*100
    open_list = []

    # for the point in the close map, they should not set their parent to curpoint
    close_map_points = []
    close_map_parents = []

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
    # use breseham
    def line_of_sight(p, p_):
        offset_x = abs(p.col_ - p_.col_)
        offset_y = abs(p.row_ - p_.row_)

        slope = (p.row_ - p_.row_) / (p.col_ - p_.col_)

        if offset_x >= offset_y:
            if p.col_ <= p_.col_:
                for i in range(1, p_.col_ - p.col_):
                    if origin_map[p.row_ + round(i*slope)][p.col_ + i] == OBSTACLE_VALUE:
                        return False
            else:
                for i in range(1, p.col_ - p_.col_):
                    if origin_map[p_.row_ + round(i*slope)][p_.col_ + i] == OBSTACLE_VALUE:
                        return False
        if offset_y >= offset_x:
            if p.row_ <= p_.row_:
                for i in range(1, p_.row_ - p.row_):
                    if origin_map[p.row_ + i][p.col_ + round(i/slope)] == OBSTACLE_VALUE:
                        return False
            else:
                for i in range(1, p.row_ - p_.row_):
                    if origin_map[p_.row_ + i][p_.col_ + round(i/slope)] == OBSTACLE_VALUE:
                        return False
       
        return True
    def set_vertex(p):
        print("==========set_vertex:", p, p.parent_)

        if p.parent_ is not None and not line_of_sight(p.parent_, p):
            min_cost = sys.maxsize
            min_point = None
            for p_ in nghbr_vis(p):

                if  close_map[p_.row_, p_.col_] != 1:
                    continue

                cost = gfunc_map[p_.row_, p_.col_] + distance(p, p_)
                print("cost:",cost)
                if cost < min_cost:
                    min_cost = cost
                    min_point = p_
                print("p_:", p_) 
                print("cost:",cost)
            # min_point.parent_.col_ = parent_mapx[point.col_]
            # min_point.parent_.row_ = parent_mapy[point.row_]
            index = close_map_points.index((min_point.row_, min_point.col_))
            min_point.parent_ = close_map_parents[index]
            p.parent_ = min_point
            print(cost, p.row_, p.col_, p.parent_.row_ , p.parent_.col_)
            gfunc_map[p.row_][p.col_] = min_cost
        # if p.parent_.parent_ is not None:

        print("set_vertex:", p, p.parent_)

    def compute_cost(p, p_):
        # print(p, p_, p.parent_)
        if p.parent_ is None:
            if gfunc_map[p.row_, p.col_] + distance(p, p_) < gfunc_map[p_.row_, p_.col_]:

                p_.parent_ = p
                gfunc_map[p_.row_, p_.col_] = gfunc_map[p.row_, p.col_] + distance(p, p_)
        else:# and line_of_sight(p.parent_, p_) :
            if gfunc_map[p.parent_.row_, p.parent_.col_] + distance(p.parent_, p_) < \
                gfunc_map[p_.row_, p_.col_]:

                p_.parent_ = p.parent_
                gfunc_map[p_.row_, p_.col_] = \
                    gfunc_map[p.parent_.row_, p.parent_.col_] + distance(p.parent_, p_)



    def update_vertex(p, p_):
        # print("update_vertex:", p, p_)

        g_old = gfunc_map[p_.row_, p_.col_]
        compute_cost(p, p_)
        print("compute_cost:", p, p_)

        # print(p_, p_.parent_)
        if gfunc_map[p_.row_, p_.col_] < g_old:
            # if open_map[p_.row_, p_.col_] == 1:
                # open_map[p_.row_, p_.col_] = 0
            if p_ in open_list:
                # open_list.remove(p_)
                # print("remove:", p_)
                [open_list.remove(p_) for p in open_list if p.row_==p_.row_ and p.col_==p_.col_]
            open_list.append((p_, gfunc_map[p_.row_, p_.col_] + h(p_)))
            open_map[p_.row_, p_.col_] = 1
        # print("len:", len(open_list), sum(open_map.flatten()))


    start_point.parent = start_point

    gfunc_map[start_point.row_][start_point.col_] = 0
    open_list.append((start_point, gfunc_map[start_point.row_][start_point.col_]+ h(start_point)))

    i = 0
    while (len(open_list) > 0):
        point, cost_value = open_list.pop(0)
        set_vertex(point)

        if point.row_ == goal_point.row_ and point.col_ == goal_point.col_:
            return point

        close_map[point.row_, point.col_] = 1


        close_map_points.append((point.row_, point.col_))
        close_map_parents.append(point.parent_)


        for p in nghbr_vis(point):

            if close_map[p.row_, p.col_] != 1:
                if open_map[p.row_, p.col_] != 1:
                # if p not in open_list:
                    gfunc_map[p.row_][p.col_] = sys.maxsize
                    p.parent_ = None
                update_vertex(point, p)
        i = i + 1
        # if i > 6:
        #     exit(0)


    return Point(sys.maxsize, sys.maxsize, None)

start_point, goal_point, origin_map = generate_map(MAP_WIDTH, MAP_HEIGHT)

print(origin_map)
end_point = LazyThetaStar(start_point, goal_point, origin_map)
print(end_point)
print(end_point.parent_)
# print(end_point.parent_.parent_)
# print(end_point.parent_.parent_.parent_)
# print(end_point.parent_.parent_.parent_.parent_)

if end_point.parent_ == None:
    print("Not Found")
    exit(0)

end_point = end_point.parent_
while end_point.row_ != start_point.row_ or end_point.col_ != start_point.col_:
    origin_map[end_point.row_][end_point.col_] = 2
    end_point = end_point.parent_
    print("end_point:", end_point)

print(origin_map)
exit(0)
