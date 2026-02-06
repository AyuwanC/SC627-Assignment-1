import heapq
import math

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_cost = {start: 0}

    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            break

        for dr, dc in moves:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            if dr != 0 and dc != 0:
                if grid[current[0]+dr, current[1]] == 1 or \
                   grid[current[0], current[1]+dc] == 1:
                    continue

            cost = math.hypot(dr, dc)
            new_g = g_cost[current] + cost
            if (nr, nc) not in g_cost or new_g < g_cost[(nr, nc)]:
                g_cost[(nr, nc)] = new_g
                h = math.hypot(goal[0]-nr, goal[1]-nc)
                heapq.heappush(open_list, (new_g + h, (nr, nc)))
                came_from[(nr, nc)] = current

    if goal not in came_from:
        raise RuntimeError("A* failed")

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path
