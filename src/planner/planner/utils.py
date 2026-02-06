import cv2
import yaml
import numpy as np

def load_map(yaml_file):
    with open(yaml_file, 'r') as f:
        map_data = yaml.safe_load(f)

    img = cv2.imread(map_data['image'], cv2.IMREAD_GRAYSCALE)
    img = np.flipud(img)

    resolution = map_data['resolution']
    origin = map_data['origin']

    # Free = 0, Obstacle = 1
    grid = np.zeros(img.shape, dtype=np.uint8)
    grid[img < 240] = 1

    return grid, resolution, origin
def world_to_grid(x, y, origin, resolution):
    gx = int((x - origin[0]) / resolution)
    gy = int((y - origin[1]) / resolution)
    return gx, gy
def nearest_free(grid, cell):
    x, y = cell
    for r in range(1, 10):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid.shape[0] and
                    0 <= ny < grid.shape[1] and
                    grid[nx][ny] == 0):
                    return (nx, ny)
    raise RuntimeError("No free cell near start/goal")
