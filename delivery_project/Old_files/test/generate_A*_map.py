import os
import numpy as np
import cv2
from isaacgym import gymapi

def generate_dynamic_gym_map(gym, sim, env, resolution=0.05, exclude_names=["floor", "carter", "tray"]):
    """
    기존 OpenCV/YAML 저장 형식을 유지하되, 
    Isaac Gym에 배치된 액터들을 기반으로 동적 맵을 생성합니다.
    """
    # 1. 맵의 전체 범위(Boundaries)를 정적 액터들의 위치를 통해 동적으로 추적
    # (예: 외벽들의 위치를 조사하여 맵의 실제 가로/세로 길이를 알아냅니다)
    actor_count = gym.get_actor_count(env)
    
    x_positions = []
    y_positions = []
    
    for i in range(actor_count):
        actor_name = gym.get_actor_name(env, i)
        if "wall" in actor_name.lower() or "pillar" in actor_name.lower():
            actor_handle = gym.get_actor_handle(env, i)
            rb_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
            if len(rb_states) > 0:
                x_positions.append(rb_states['pose']['p'][0]['x'])
                y_positions.append(rb_states['pose']['p'][0]['y'])
    
    # 외곽 기둥/벽 좌표 기반 가로 세로 계산 (조금의 마진 제공)
    if len(x_positions) > 0:
        WIDTH_M = (max(x_positions) - min(x_positions)) + 0.4  # 기둥 두께 포함
        HEIGHT_M = (max(y_positions) - min(y_positions)) + 0.4
    else:
        # 폴백(Fallback) 기본값
        WIDTH_M, HEIGHT_M = 6.75, 12.0

    width_px = int(WIDTH_M / resolution)
    height_px = int(HEIGHT_M / resolution)

    # 맵 배열 초기화
    grid_map = np.zeros((height_px, width_px), dtype=np.uint8)

    # 기존 world_to_grid 구조 유지
    def world_to_grid(x, y):
        px = int((x + WIDTH_M / 2) / resolution)
        py = int((HEIGHT_M / 2 - y) / resolution)  # OpenCV 이미지 처리를 위한 Y축 반전 유지
        return px, py

    def draw_obstacle(cx, cy, w, h):
        px, py = world_to_grid(cx, cy)
        pw = int(w / resolution / 2)
        ph = int(h / resolution / 2)
        
        x1, x2 = max(0, px - pw), min(width_px, px + pw)
        y1, y2 = max(0, py - ph), min(height_px, py + ph)
        grid_map[y1:y2, x1:x2] = 1

    # 2. 모든 액터를 순회하며 동적으로 장애물 마킹
    for i in range(actor_count):
        actor_name = gym.get_actor_name(env, i)
        
        # 제외할 객체(주행할 로봇 등)는 패스
        if any(exclude in actor_name.lower() for exclude in exclude_names):
            continue
            
        actor_handle = gym.get_actor_handle(env, i)
        rb_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
        if len(rb_states) == 0:
            continue
            
        cx = rb_states['pose']['p'][0]['x']
        cy = rb_states['pose']['p'][0]['y']
        
        # 액터 이름별 규격 분기 (에셋 규격 동적 매핑)
        obs_w, obs_h = 0.4, 0.4  # 기본값 (기둥)
        
        if "wall" in actor_name.lower():
            if "wall_back" in actor_name or "wall_front" in actor_name:
                obs_w, obs_h = 0.2, HEIGHT_M
            elif "wall_right" in actor_name or "wall_left" in actor_name:
                obs_w, obs_h = WIDTH_M, 0.2
        elif "shelf" in actor_name.lower():
            obs_w, obs_h = 1.3, 1.3  # ㄷ자 선반 규격
        elif "franka" in actor_name.lower():
            obs_w, obs_h = 0.6, 0.6  # 프랑카 베이스
            
        # 장애물 그리기 실행
        draw_obstacle(cx, cy, obs_w, obs_h)

    # 3. 기존 출력 및 저장 로직 완벽 유지
    os.makedirs("map", exist_ok=True)

    npy_path = "map/map.npy"
    np.save(npy_path, grid_map)

    img_map = np.where(grid_map == 1, 0, 255).astype(np.uint8)
    cv2.imwrite("map/map.png", img_map)

    yaml_path = "map/map.yaml"
    with open(yaml_path, "w") as f:
        f.write("image: map.png\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"origin: [{-WIDTH_M/2}, {-HEIGHT_M/2}, 0.0]\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
        f.write("negate: 0\n")

    print(f"\n[동적 스캔 성공] 현재 Gym 레이아웃 맞춤형 맵 생성 완료.")
    print(f" - 계산된 맵 크기: {WIDTH_M}m x {HEIGHT_M}m ({width_px}x{height_px} px)")
    print(f" - 저장 경로: {npy_path}, {yaml_path}")
    
    return grid_map