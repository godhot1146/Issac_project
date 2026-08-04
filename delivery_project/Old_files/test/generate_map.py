import numpy as np
import cv2
import os

# 1. 맵 해상도 및 규격 설정 (main.py와 동일하게 맞춤)
RESOLUTION = 0.05  # 1픽셀당 0.05m (5cm)
WIDTH_M = 6.75     # X축 가로 길이
HEIGHT_M = 12.0    # Y축 세로 길이

width_px = int(WIDTH_M / RESOLUTION)
height_px = int(HEIGHT_M / RESOLUTION)

# 맵 배열 초기화 (0: 안전한 빈 공간, 1: 이동 불가 장애물)
grid_map = np.zeros((height_px, width_px), dtype=np.uint8)

def world_to_grid(x, y):
    """실제 3D 좌표(m)를 2D 이미지 픽셀 좌표로 변환"""
    # 중심 좌표(0,0)를 이미지의 정중앙 픽셀로 이동 (Y축은 위에서 아래로 내려가므로 반전)
    px = int((x + WIDTH_M / 2) / RESOLUTION)
    py = int((HEIGHT_M / 2 - y) / RESOLUTION)
    return px, py

def draw_obstacle(cx, cy, w, h):
    """중심 좌표(cx, cy)에 너비(w), 높이(h)만큼의 사각형 장애물을 그림"""
    px, py = world_to_grid(cx, cy)
    pw = int(w / RESOLUTION / 2)
    ph = int(h / RESOLUTION / 2)
    
    # 맵 배열 인덱스를 벗어나지 않도록 클리핑
    x1, x2 = max(0, px - pw), min(width_px, px + pw)
    y1, y2 = max(0, py - ph), min(height_px, py + ph)
    
    grid_map[y1:y2, x1:x2] = 1  # 장애물 영역을 1로 마킹

# 2. 장애물 정보 입력 (main.py의 좌표와 완벽히 동기화)
# ① 외벽 두께 (0.2m)
grid_map[0:int(0.2/RESOLUTION), :] = 1
grid_map[-int(0.2/RESOLUTION):, :] = 1
grid_map[:, 0:int(0.2/RESOLUTION)] = 1
grid_map[:, -int(0.2/RESOLUTION):] = 1

# ② ㄷ자형 선반 배치 (선반 규격 약 1.3m x 1.3m)
shelf_positions = [
    [-2.3, 3.49], [-2.3, 2.13], [-2.3, 0.77], [-2.3, -0.59],
    [-2.3, 5.0], [-0.5, 5.0], [1.0, 5.0], [2.5, 5.0],
    [2.3, 3.49], [2.3, 2.13], [2.3, 0.77], [2.3, -0.59], [2.3, -1.95], [2.3, -3.31], [2.3, -4.76]
]
for pos in shelf_positions:
    draw_obstacle(pos[0], pos[1], 1.3, 1.3)

# ③ Franka 로봇 베이스 (안전 반경 포함 0.6m x 0.6m)
draw_obstacle(-1.0, 0.0, 0.6, 0.6)

# ④ 코너 및 사이드 기둥 (0.4m x 0.4m)
p_offset_w = WIDTH_M / 2 - 0.25
p_offset_l = HEIGHT_M / 2 - 0.25
pillar_side_positions = np.arange(-2.5, 3.5, 2.5)

for x in [-p_offset_w, p_offset_w]:
    for y in [-p_offset_l, p_offset_l]:
        draw_obstacle(x, y, 0.4, 0.4)
        
for y_pos in pillar_side_positions:
    for x_pos in [-p_offset_w, p_offset_w]:
        draw_obstacle(x_pos, y_pos, 0.4, 0.4)

# 3. 맵 파일 저장
os.makedirs("map", exist_ok=True)

# A* 플래너가 직접 연산할 넘파이 배열
npy_path = "map/map.npy"
np.save(npy_path, grid_map)

# 시각적 확인용 이미지 (장애물은 검은색(0), 빈 공간은 흰색(255))
img_map = np.where(grid_map == 1, 0, 255).astype(np.uint8)
cv2.imwrite("map/map.png", img_map)

# ROS2 Nav2 규격의 메타데이터 yaml 파일 고정 이름으로 생성
yaml_path = "map/map.yaml"
with open(yaml_path, "w") as f:
    f.write("image: map.png\n")
    f.write(f"resolution: {RESOLUTION}\n")
    f.write(f"origin: [{-WIDTH_M/2}, {-HEIGHT_M/2}, 0.0]\n")
    f.write("occupied_thresh: 0.65\n")
    f.write("free_thresh: 0.196\n")
    f.write("negate: 0\n")

print(f"\n[성공] 새로운 환경 맵이 성공적으로 생성되었습니다.")
print(f" - A* 연산용 : {npy_path}")
print(f" - 시각 확인용: map/map.png")
print(f" - 메타데이터 : {yaml_path}\n")