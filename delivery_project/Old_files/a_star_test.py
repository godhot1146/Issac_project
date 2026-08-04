import numpy as np
import yaml
import heapq
import math
import os
from datetime import datetime

class AStarPlanner:
    def __init__(self, map_npy_path, map_yaml_path, robot_radius=0.4):
        raw_grid = np.load(map_npy_path)
        raw_grid = np.rot90(raw_grid, k=1)
        
        with open(map_yaml_path, 'r') as f:
            self.map_info = yaml.safe_load(f)
        # print(raw_grid.shape[0])
        self.resolution = 16.0 / raw_grid.shape[1]
        self.origin = [-8.0, -5.0]
        self.robot_radius = robot_radius
        
        self.grid = np.zeros_like(raw_grid, dtype=np.float32)
        grid_max = np.max(raw_grid)
        
        if grid_max > 100:
            self.grid[raw_grid < 128] = 100  
            self.grid[raw_grid >= 128] = 0   
        else:
            threshold = grid_max / 2.0 if grid_max > 0 else 0.5
            self.grid[raw_grid > threshold] = 100  
            self.grid[raw_grid <= threshold] = 0   
            
        self.inflation_cells = int(math.ceil(robot_radius / self.resolution))
        self.grid_inflated = self._inflate_map(self.grid, self.inflation_cells)

    def _inflate_map(self, grid, inflation_cells):
        inflated = np.copy(grid)
        rows, cols = grid.shape
        obstacle_coords = np.argwhere(grid > 50)  
        
        for r, c in obstacle_coords:
            r_min = max(0, r - inflation_cells)
            r_max = min(rows, r + inflation_cells + 1)
            c_min = max(0, c - inflation_cells)
            c_max = min(cols, c + inflation_cells + 1)
            inflated[r_min:r_max, c_min:c_max] = 100
        return inflated

    def world_to_grid(self, x, y):
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        
        rows, cols = self.grid.shape
        row = (rows - 1) - row
        
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))
        
        return (row, col)

    def grid_to_world(self, row, col):
        rows = self.grid.shape[0]
        row_flipped = (rows - 1) - row
        
        x = col * self.resolution + self.origin[0]
        y = row_flipped * self.resolution + self.origin[1]
        return (x, y)

    def _calculate_docking_cost(self, current, neighbor, goal_node):
        if self.robot_radius > 0.3:
            return 0.0  

        goal_world = self.grid_to_world(goal_node[0], goal_node[1])
        if not (1.8 <= goal_world[0] <= 2.2 and 0.7 <= goal_world[1] <= 0.8):
            return 0.0  

        neighbor_world = self.grid_to_world(neighbor[0], neighbor[1])
        if 1.1 <= neighbor_world[0] <= 2.0:
            dr = neighbor[0] - current[0]
            row_deviation = abs(neighbor[0] - goal_node[0])

            if dr != 0 or row_deviation > 0:
                return 5000.0  

        return 0.0

    # ★ [인터페이스 완전 일치 보장] save_debug_img 매개변수를 명시적으로 선언
    def plan_path(self, start_world, goal_world, save_debug_img=True):
        start_node = self.world_to_grid(start_world[0], start_world[1])
        node_goal = self.world_to_grid(goal_world[0], goal_world[1])

        rows, cols = self.grid_inflated.shape
        
        print(f"[A* Planner] Start World: {start_world} -> Grid: {start_node}")
        print(f"[A* Planner] Goal World: {goal_world} -> Grid: {node_goal}")

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        
        final_path = []

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == node_goal:
                while current in came_from:
                    final_path.append(self.grid_to_world(current[0], current[1]))
                    current = came_from[current]
                final_path.reverse()
                break

            for dr, dc in neighbors:
                neighbor = (current[0] + dr, current[1] + dc)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    if self.grid_inflated[neighbor] > 50: 
                        continue
                    
                    base_cost = 1.414 if dr != 0 and dc != 0 else 1.0
                    docking_penalty = self._calculate_docking_cost(current, neighbor, node_goal)
                    tentative_g = g_score[current] + base_cost + docking_penalty

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = math.hypot(node_goal[0] - neighbor[0], node_goal[1] - neighbor[1])
                        heapq.heappush(open_set, (tentative_g + h, neighbor))
        
        ### 디버깅 용 맵 그리기
        if save_debug_img:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 8))
                plt.imshow(self.grid_inflated, cmap='gray_r')
                
                # 시작점(Blue)과 목적지(Green)는 무조건 맵 위에 드로잉
                plt.plot(start_node[1], start_node[0], 'bo', markersize=8, label='Start')
                plt.plot(node_goal[1], node_goal[0], 'go', markersize=8, label='Goal')
                
                # 경로 탐색에 성공했을 때만 빨간색 선(Path)을 추가로 드로잉
                if final_path:
                    path_r = [self.world_to_grid(p[0], p[1])[0] for p in final_path]
                    path_c = [self.world_to_grid(p[0], p[1])[1] for p in final_path]
                    plt.plot(path_c, path_r, 'r-', linewidth=2, label='Path')
                    plt.title("A* Map Debugging - Success")
                else:
                    plt.title("A* Map Debugging - PATH FAILURE")
                    
                plt.legend()
                
                base_dir = os.path.dirname(os.path.abspath(__file__))
                debug_dir = os.path.join(base_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 파일명에 성공/실패 여부를 접미사로 붙여 가시성 확보
                status_str = "success" if final_path else "FAIL"
                filename = f"debug_map_{timestamp}_{status_str}.png"
                save_path = os.path.join(debug_dir, filename)
                
                plt.savefig(save_path)
                plt.close()
                print(f">>> 디그 맵 이미지가 저장되었습니다: {save_path}")
            except Exception as e:
                print(">>> 디그 이미지 저장 실패:", e)
        ###

        if not final_path:
            print("에러: 경로를 찾을 수 없습니다.")
            return final_path
            
        # 루프 내부 실시간 연산 시에는 이미지 드로잉 세션을 스킵하여 BadWindow 폭파 차단
        if not save_debug_img:
            return final_path

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(self.grid_inflated, cmap='gray_r')
            plt.plot(start_node[1], start_node[0], 'bo', markersize=8, label='Start')
            plt.plot(node_goal[1], node_goal[0], 'go', markersize=8, label='Goal')
            
            if final_path:
                path_r = [self.world_to_grid(p[0], p[1])[0] for p in final_path]
                path_c = [self.world_to_grid(p[0], p[1])[1] for p in final_path]
                plt.plot(path_c, path_r, 'r-', linewidth=2, label='Path')
                
            plt.legend()
            plt.title("A* Map Debugging")
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            debug_dir = os.path.join(base_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_map_{timestamp}.png"
            save_path = os.path.join(debug_dir, filename)
            
            plt.savefig(save_path)
            plt.close()
            print(f">>> 디버그 맵 이미지가 저장되었습니다: {save_path}")
        except Exception as e:
            print(">>> 디버그 이미지 저장 실패:", e)

        return final_path