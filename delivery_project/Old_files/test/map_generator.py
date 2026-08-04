import os
import numpy as np
import yaml
import matplotlib.pyplot as plt

class AStarMapGenerator:
    def __init__(self, room_width, room_length, resolution, wall_thickness):
        """
        맵 생성을 처리하기 위한 초기화 매개변수 설정
        """
        self.room_width = room_width
        self.room_length = room_length
        self.resolution = resolution
        self.wall_thickness = wall_thickness
        
        # 픽셀 단위 해상도 크기 계산
        self.map_width_pixels = int(self.room_width / self.resolution)
        self.map_length_pixels = int(self.room_length / self.resolution)
        
        # 빈 그리드 맵 초기화 (0: 이동 가능, 1: 장애물)
        self.grid_map = np.zeros((self.map_length_pixels, self.map_width_pixels), dtype=np.uint8)

    def world_to_grid(self, x, y):
        """월드 좌표(m)를 그리드 맵의 인덱스(Pixel)로 변환"""
        col = int((x + self.room_width / 2.0) / self.resolution)
        row = int((y + self.room_length / 2.0) / self.resolution)
        return self._clamp(row, 0, self.map_length_pixels - 1), self._clamp(col, 0, self.map_width_pixels - 1)

    def _clamp(self, n, minn, maxn):
        return max(min(n, maxn), minn)

    def mark_walls(self):
        """외벽 두께를 반영하여 테두리를 장애물로 마킹"""
        wall_pixel_thickness = int(self.wall_thickness / self.resolution)
        
        # 상/하/좌/우 외곽 영역 차단
        self.grid_map[0:wall_pixel_thickness, :] = 1
        self.grid_map[-wall_pixel_thickness:, :] = 1
        self.grid_map[:, 0:wall_pixel_thickness] = 1
        self.grid_map[:, -wall_pixel_thickness:] = 1

    def mark_rectangular_obstacles(self, positions, obstacle_width, obstacle_length):
        """선반이나 기둥 같은 직사각형 장애물들을 배치 처리"""
        width_pixels = int((obstacle_width / 2.0) / self.resolution)
        length_pixels = int((obstacle_length / 2.0) / self.resolution)
        
        for pos in positions:
            center_row, center_col = self.world_to_grid(pos[0], pos[1])
            
            r_start = max(0, center_row - length_pixels)
            r_end = min(self.map_length_pixels, center_row + length_pixels)
            c_start = max(0, center_col - width_pixels)
            c_end = min(self.map_width_pixels, center_col + width_pixels)
            
            self.grid_map[r_start:r_end, c_start:c_end] = 1

    def save_map(self, output_dir, npy_filename, yaml_filename, png_filename=None):
        """생성된 맵 데이터를 .npy, .yaml 그리고 시각화 .png 이미지로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        npy_path = os.path.join(output_dir, npy_filename)
        yaml_path = os.path.join(output_dir, yaml_filename)
        
        # 1. binary 맵 데이터 저장
        np.save(npy_path, self.grid_map)
        
        # 2. 메타데이터 YAML 작성
        meta_data = {
            'image': npy_filename,
            'resolution': self.resolution,
            'origin': [-self.room_width / 2.0, -self.room_length / 2.0, 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196
        }
        
        if png_filename:
            meta_data['negate'] = 0  # 이미지 매핑용 옵션 유지
            
        with open(yaml_path, 'w') as f:
            yaml.dump(meta_data, f)
            
        # 3. ★ 추가: 맵의 형태를 한눈에 볼 수 있는 PNG 이미지 시각화 저장
        if png_filename:
            png_path = os.path.join(output_dir, png_filename)
            
            plt.figure(figsize=(self.room_width, self.room_length))
            # 0(흰색: 주행가능), 1(검은색: 장애물) 반전하여 직관적인 binary map 표현
            plt.imshow(1 - self.grid_map, cmap='gray', origin='lower')
            plt.axis('off')  # 외곽 축 숨기기
            
            # 이미지 여백 최소화하여 크롭 저장
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            print(f"[맵 제너레이터] 모니터링용 사진 파일이 저장되었습니다. -> {png_path}")
            
        print(f"[맵 제너레이터] A* 지도가 성공적으로 빌드되었습니다. 경로: {output_dir}")