import os
import numpy as np
import yaml
import heapq
import math
from datetime import datetime

class AStarPlanner:
    def __init__(self, map_npy_path, map_yaml_path, robot_radius=0.4):
        with open(map_yaml_path, 'r') as f:
            self.map_info = yaml.safe_load(f)

        self.resolution = float(self.map_info['resolution'])
        self.robot_radius = robot_radius

        # 🚀 하드코딩 제거: 룸의 크기를 메타데이터나 월드 스케일 기준으로 역산
        # origin이 [-width/2, -length/2]로 들어오므로 절대값을 두 배 하면 룸 크기가 됩니다.
        self.room_width = abs(float(self.map_info['origin'][0])) * 2.0
        self.room_length = abs(float(self.map_info['origin'][1])) * 2.0

        # 원본 그리드 로드 (상하/좌우 반전 없이 마우스 PNG와 100% 일치 상태 유지)
        self.grid_inflated = np.load(map_npy_path)
        self.rows, self.cols = self.grid_inflated.shape

        # 로봇 반지름 기반 장애물 팽창(Inflation) 셀 크기 계산
        self.inflation_cells = int(math.ceil(self.robot_radius / self.resolution))
        self._inflate_map_inline()

    def _inflate_map_inline(self):
        """기존 맵 기반 장애물 팽창 영역 적용"""
        obstacle_coords = np.argwhere(self.grid_inflated > 0)  
        for r, c in obstacle_coords:
            r_min = max(0, r - self.inflation_cells)
            r_max = min(self.rows, r + self.inflation_cells + 1)
            c_min = max(0, c - self.inflation_cells)
            c_max = min(self.cols, c + self.inflation_cells + 1)
            self.grid_inflated[r_min:r_max, c_min:c_max] = 1

    # 🚀 마우스 클릭 변환식과 100% 동일하게 일치시킨 월드 -> 그리드 변환
    def world_to_grid(self, x, y):
        """마우스 클릭의 비율 계산식과 매커니즘 완전 일치"""
        col = int(((x + self.room_width / 2.0) / self.room_width) * self.cols)
        row = int(((y + self.room_length / 2.0) / self.room_length) * self.rows)
        
        # 인덱스 바운딩 제어
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return (row, col)

    # 🚀 마우스 클릭 변환식과 100% 동일하게 일치시킨 그리드 -> 월드 변환
    def grid_to_world(self, row, col):
        """그리드 인덱스(Pixel)를 월드 미터(m) 좌표로 정밀 역산"""
        x = (col / self.cols) * self.room_width - (self.room_width / 2.0)
        y = (row / self.rows) * self.room_length - (self.room_length / 2.0)
        return (x, y)

    def _calculate_docking_cost(self, current, neighbor, goal_node):
        if self.robot_radius > 0.3:
            return 0.0  
        goal_world = self.grid_to_world(goal_node[0], goal_node[1])
        dist_to_goal = math.hypot(goal_world[0] - self.grid_to_world(neighbor[0], neighbor[1])[0],
                                  goal_world[1] - self.grid_to_world(neighbor[0], neighbor[1])[1])
        if dist_to_goal < 0.6:
            dr = neighbor[0] - current[0]
            dc = neighbor[1] - current[1]
            if dr != 0 and dc != 0:
                return 1000.0
        return 0.0

    def plan_path(self, start_world, goal_world, robot_name="robot", save_debug_img=True):
        # 개조된 직관적 변환식 호출
        start_node = self.world_to_grid(start_world[0], start_world[1])
        node_goal = self.world_to_grid(goal_world[0], goal_world[1])

        print(f"[{robot_name.upper()} A* 플래닝 가동 - 마우스 좌표계 기준 동기화 버전]")
        print(f" -> Start Grid: {start_node} | Goal Grid: {node_goal}")

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        final_path = []

        # 진입 검증
        if self.grid_inflated[start_node] > 0:
            print(f"⚠️ 경고: {robot_name} 시작점이 팽창 장애물 영역 내부입니다.")
        if self.grid_inflated[node_goal] > 0:
            print(f"⚠️ 에러: {robot_name} 목적지가 장애물 영역 내부입니다. 패스 생성 불가.")
            self._save_debug_image(start_node, node_goal, final_path, robot_name, save_debug_img)
            return final_path

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
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    if self.grid_inflated[neighbor] > 0: 
                        continue
                    
                    base_cost = 1.414 if dr != 0 and dc != 0 else 1.0
                    docking_penalty = self._calculate_docking_cost(current, neighbor, node_goal)
                    tentative_g = g_score[current] + base_cost + docking_penalty

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = math.hypot(node_goal[0] - neighbor[0], node_goal[1] - neighbor[1])
                        heapq.heappush(open_set, (tentative_g + h, neighbor))
        
        self._save_debug_image(start_node, node_goal, final_path, robot_name, save_debug_img)
        return final_path

    def _save_debug_image(self, start_node, node_goal, final_path, robot_name, save_debug_img):
        if not save_debug_img:
            return
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 10))
            # 마우스 창과 100% 일치하도록 데이터 드로잉 축 고정
            plt.imshow(self.grid_inflated, cmap='gray_r', origin='lower')
            
            # Matplotlib은 가로가 X(Col), 세로가 Y(Row)이므로 인덱스 순서 주의 [Col, Row]
            plt.plot(start_node[1], start_node[0], 'bo', markersize=8, label='Start')
            plt.plot(node_goal[1], node_goal[0], 'go', markersize=8, label='Goal')
            
            if final_path:
                path_r = [self.world_to_grid(p[0], p[1])[0] for p in final_path]
                path_c = [self.world_to_grid(p[0], p[1])[1] for p in final_path]
                plt.plot(path_c, path_r, 'r-', linewidth=2, label='Path')
            
            plt.legend()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            debug_dir = os.path.join(base_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            filename = f"debug_map_{robot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            print(filename)
            plt.savefig(os.path.join(debug_dir, filename), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print("디버그 이미지 저장 실패:", e)