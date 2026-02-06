import cv2
import numpy as np
import heapq
from collections import deque

# ---------- Distance Transform ----------

def distance_transform(grid):
    obstacle_img = (grid * 255).astype(np.uint8)
    return cv2.distanceTransform(255 - obstacle_img, cv2.DIST_L2, 5)


# ---------- GVD REGION (tunable) ----------

def extract_gvd_region(grid, dist, clearance_cells):
    """
    Approximate GVD as high-clearance free space
    """
    gvd = np.zeros_like(grid)
    gvd[(grid == 0) & (dist >= clearance_cells)] = 1
    return gvd


# ---------- Phase 1: Attach start/goal to GVD ----------

def attach_to_gvd(grid, gvd, start):
    q = deque([start])
    came_from = {start: None}

    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        cur = q.popleft()
        if gvd[cur] == 1:
            return reconstruct(came_from, cur)

        for dr, dc in moves:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if grid[nr,nc] == 1:
                continue
            if (nr,nc) in came_from:
                continue

            came_from[(nr,nc)] = cur
            q.append((nr,nc))

    raise RuntimeError("Cannot attach to GVD")


# ---------- Phase 2: Clearance-guided traversal ----------

def traverse_gvd(gvd, dist, start, goal):
    pq = []
    heapq.heappush(pq, (-dist[start], start))
    came_from = {start: None}
    visited = set()

    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            return reconstruct(came_from, goal)

        if cur in visited:
            continue
        visited.add(cur)

        for dr, dc in moves:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < gvd.shape[0] and 0 <= nc < gvd.shape[1]):
                continue
            if gvd[nr,nc] != 1:
                continue
            if (nr,nc) in visited:
                continue

            if (nr,nc) not in came_from:
                came_from[(nr,nc)] = cur
                heapq.heappush(pq, (-dist[nr,nc], (nr,nc)))

    raise RuntimeError("GVD traversal failed")


# ---------- Utilities ----------

def reconstruct(came_from, end):
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]
