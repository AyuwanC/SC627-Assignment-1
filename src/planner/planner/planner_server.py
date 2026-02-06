import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point
from planner.utils import load_map, nearest_free, world_to_grid
from planner.astar import astar
from planner.gvd import *
import numpy as np

class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server')

        self.a_pub = self.create_publisher(Path, '/a_star_path', 10)
        self.s_pub = self.create_publisher(Path, '/safe_path', 10)
        self.timer = self.create_timer(0.05, self.animate_paths)
        self.debug_grid_pub = self.create_publisher(
            OccupancyGrid, '/debug_grid', 1
        )

        grid, res, origin = load_map('src/planner/planner/gauntlet.yaml')

        start = world_to_grid(-1.1, -0.8, origin, res)
        goal  = world_to_grid( 1.1,  1.1, origin, res)
        start = nearest_free(grid, start)
        goal  = nearest_free(grid, goal)


        # A*
        self.publish_debug_grid(grid, res, origin)

        a_path = astar(grid, start, goal)
        self.a_path = a_path

        #gvd
        dist = distance_transform(grid)

        CLEARANCE_CELLS = 3 # can tune
        gvd = extract_gvd_region(grid, dist, CLEARANCE_CELLS)

        # viz for gvd grid
        # self.publish_debug_grid(gvd, res, origin)

        #attach start and goal
        path_start = attach_to_gvd(grid, gvd, start)
        path_goal  = attach_to_gvd(grid, gvd, goal)

        gvd_start = path_start[-1]
        gvd_goal  = path_goal[-1]

        #medial region
        path_mid = traverse_gvd(gvd, dist, gvd_start, gvd_goal)

        #safe path me start and end add karo
        self.s_path = path_start[:-1] + path_mid + path_goal[::-1][1:]
        self.a_idx = 0
        self.s_idx = 0

        self.resolution = res
        self.origin = origin

    def animate_paths(self):
        # animation for A*
        if self.a_idx < len(self.a_path):
            msg = Path()
            msg.header.frame_id = 'map'

            for r, c in self.a_path[:self.a_idx + 1]:
                p = PoseStamped()
                p.pose.position.x = self.origin[0] + c * self.resolution
                p.pose.position.y = self.origin[1] + r * self.resolution
                p.pose.orientation.w = 1.0
                msg.poses.append(p)


            self.a_pub.publish(msg)
            self.a_idx += 1
            return

        # Animation for gvd
        # The colour of the path and the line thickness has to be changes in RViz only as Path does not have any control over color unlike Marker
        if self.s_idx < len(self.s_path):
            msg = Path()
            msg.header.frame_id = 'map'

            for r, c in self.s_path[:self.s_idx + 1]:
                p = PoseStamped()
                p.pose.position.x = self.origin[0] + c * self.resolution
                p.pose.position.y = self.origin[1] + r * self.resolution
                p.pose.orientation.w = 1.0
                msg.poses.append(p)

            self.s_pub.publish(msg)
            self.s_idx += 1
    def publish_debug_grid(self, grid, resolution, origin):
        msg = OccupancyGrid()

        msg.header.frame_id = 'map'
        msg.info.resolution = resolution
        msg.info.width = grid.shape[1]   # cols
        msg.info.height = grid.shape[0]  # rows
        msg.info.origin.position.x = origin[0]
        msg.info.origin.position.y = origin[1]
        msg.info.origin.orientation.w = 1.0

        # ROS OccupancyGrid: for debugging as I had the issue of the goal being unreachable so the gvd bfs would keep failing
        data = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 1:
                    data.append(100)
                else:
                    data.append(0)

        msg.data = data
        self.debug_grid_pub.publish(msg)

def main():
    rclpy.init()
    node = PlannerServer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
