# import os
# import numpy as np
# import yaml
# import matplotlib.pyplot as plt

# def visualize_Isaac_gym_map():
#     # 1. 경로 설정
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     map_dir = os.path.join(base_dir, "map")
#     npy_path = os.path.join(map_dir, "map.npy")
#     yaml_path = os.path.join(map_dir, "map.yaml")
    
#     if not os.path.exists(npy_path) or not os.path.exists(yaml_path):
#         print(f"에러: 맵 파일이 {map_dir}에 존재하지 않습니다.")
#         return

#     # 2. 파일 로드
#     raw_grid = np.load(npy_path)
#     with open(yaml_path, 'r') as f:
#         map_info = yaml.safe_load(f)
        
#     print("--- YAML Map Info ---")
#     print(yaml.dump(map_info))

#     # 3. 로봇 플래너와 동일하게 이미지 방향 맞추기 (회전 및 플립)
#     # Isaac Gym과 매핑을 위해 플래너가 처리하던 rot90 가공을 동일하게 적용합니다.
#     grid = np.rot90(raw_grid, k=1)
#     rows, cols = grid.shape

#     # 4. YAML 데이터를 기반으로 물리적 실제 크기(Extent) 계산
#     # 일반적인 ROS/Isaac Gym 스타일 사양 분석
#     resolution = map_info.get('resolution', 16.0 / cols)
#     origin = map_info.get('origin', [-8.0, -8.0])
    
#     # 맵의 물리적 가로세로 m 스케일 계산
#     width_m = cols * resolution
#     height_m = rows * resolution
    
#     x_min = origin[0]
#     x_max = origin[0] + width_m
#     y_min = origin[1]
#     y_max = origin[1] + height_m

#     # 5. Matplotlib 드로잉
#     plt.figure(figsize=(10, 10))
    
#     # extent를 주면 축 눈금이 픽셀 번호가 아닌 월드 좌표(m)로 표시됩니다.
#     plt.imshow(grid, cmap='gray_r', extent=[x_min, x_max, y_min, y_max])
    
#     plt.colorbar(label="Obstacle Occupancy / Cost")
#     plt.grid(True, color='blue', alpha=0.15, linestyle='--')
    
#     plt.xlabel("World X (meters)")
#     plt.ylabel("World Y (meters)")
#     plt.title(f"Isaac Gym World Grid Map ({rows}x{cols})\nResolution: {resolution:.4f}m/px")
    
#     # 절대 원점 (0.0, 0.0) 표시
#     plt.plot(0, 0, 'ro', markersize=10, label="World Origin (0,0)")
    
#     # 앞서 실패했던 좌표들 디버깅용 점 찍기
#     plt.plot(0.0, 0.7, 'm^', markersize=8, label="AMR Start (0.0, 0.7)")
#     plt.plot(1.0, 0.7, 'mx', markersize=8, label="AMR Carter Goal (1.0, 0.7)")
#     plt.plot(1.5, 0.77, 'c^', markersize=8, label="LowProfile Start (1.5, 0.77)")
#     plt.plot(3.5, 0.77, 'cx', markersize=8, label="LowProfile Goal (3.5, 0.77)")
    
#     plt.legend(loc='upper right')
    
#     # 이미지 저장 및 출력
#     debug_dir = os.path.join(base_dir, "debug")
#     os.makedirs(debug_dir, exist_ok=True)
#     save_path = os.path.join(debug_dir, "world_map_check.png")
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"\n>>> 지도가 성공적으로 시각화되어 저장되었습니다: {save_path}")
#     plt.show()

# if __name__ == "__main__":
#     visualize_Isaac_gym_map()

import os
import numpy as np
import matplotlib.pyplot as plt

def convert_npy_to_image():
    # 1. 파일 경로 매핑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(base_dir, "map", "map.npy")
    
    if not os.path.exists(npy_path):
        print(f"에러: {npy_path} 경로에 npy 파일이 없습니다.")
        return

    # 2. npy 데이터 전체 로드 (NumPy Array)
    raw_grid = np.load(npy_path)
    
    # 플래너와 연동 규격을 맞추기 위해 90도 회전 (필요 없다면 이 줄을 주석 처리하세요)
    grid = np.rot90(raw_grid, k=1)
    
    rows, cols = grid.shape
    print(f"[데이터 확인] 배열 형태(Shape): {rows} x {cols}")
    print(f"[데이터 확인] 데이터 타입: {grid.dtype}")
    print(f"[데이터 확인] 고유값 종류: {np.unique(grid)}") # 내부에 어떤 숫자들이 들어있는지 출력

    # 3. 플로팅 세션 개시
    plt.figure(figsize=(10, 10))
    
    # gray_r 반전 컬러맵: 0(빈 공간)은 흰색, 높은 값(장애물)은 검은색/어두운색 매핑
    # origin='upper'를 주어 배열의 [0, 0] 인덱스가 왼쪽 상단(Top-Left)에 오도록 전체 출력
    plt.imshow(grid, cmap='gray_r', origin='upper')
    
    # 축 눈금을 픽셀 번호(인덱스)로 촘촘하게 표시
    plt.colorbar(label="Pixel Value (Cost)")
    plt.grid(True, color='red', alpha=0.2, linestyle=':')
    
    plt.xlabel("Grid Column Index (X)")
    plt.ylabel("Grid Row Index (Y)")
    plt.title(f"Complete Raw npy Grid Map ({rows}x{cols})")

    # 4. 이미지 파일 저장 및 시각화 종료
    debug_dir = os.path.join(base_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.join(debug_dir, "raw_npy_matrix.png")
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\n>>> npy 행렬 데이터가 전체 이미지로 변환되었습니다!")
    print(f">>> 저장 경로: {output_path}")

if __name__ == "__main__":
    convert_npy_to_image()