import os
import numpy as np
from isaacgym import gymapi, gymutil
from franka_controller import FrankaController
from carter_amr import CarterAMR
from low_amr import LowProfileAMR
from a_star_test import AStarPlanner

# 1. 시뮬레이션 및 물리 엔진 초기화
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Franka-AMR Project")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 12

sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.001
sim_params.physx.bounce_threshold_velocity = 0.2

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# ★ 은은하고 부드러운 스튜디오 조명 유지 (그림자 최소화)
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0.7, 0.7, 0.7), gymapi.Vec3(0.2, 0.3, -1.0))

# 2. 바닥 평면 생성
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)

# =================================================================
# 3. 에셋 규격 및 가상 맵 정의
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

# ★ 공장 사이즈 변경 (6.75m x 12.0m)
room_width = 6.75 # -> generate_A*_map
room_length = 12.0 # -> generate_A*_map
wall_height = 10.0
wall_thickness = 0.2 # -> generate_A*_map

pillar_x_size = 0.4  
pillar_y_size = 0.4

# static asset 바닥 및 외벽 직사각형 비율 적용
floor_asset = gym.create_box(sim, room_width, room_length, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_length, wall_height, env_opts)
wall_y = gym.create_box(sim, room_width, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, pillar_x_size, pillar_y_size, wall_height, env_opts)

shelf_opts = gymapi.AssetOptions()
shelf_opts.fix_base_link = False
shelf_opts.density = 100.0
cargo_shelf_asset = gym.load_asset(sim, asset_root, "urdf/cargo_shelf/cargo_shelf.urdf", shelf_opts)

# dynamic asset
f_opts = gymapi.AssetOptions()
f_opts.fix_base_link, f_opts.flip_visual_attachments = True, True
f_opts.armature = 0.01
franka_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/franka_panda.urdf", f_opts)

box_size = 0.22
box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = True
box_asset = gym.create_box(sim, box_size, box_size, box_size, box_opts)

c_opts = gymapi.AssetOptions()
c_opts.fix_base_link = False
carter_asset = gym.load_asset(sim, asset_root, "urdf/carter/carter.urdf", c_opts)

tray_opts = gymapi.AssetOptions()
tray_opts.fix_base_link, tray_opts.disable_gravity = False, True
tray_asset = gym.load_asset(sim, asset_root, "urdf/tray/traybox.urdf", tray_opts)

low_amr_opts = gymapi.AssetOptions()
low_amr_opts.fix_base_link = False
low_amr_asset = gym.load_asset(sim, asset_root, "urdf/low_amr/low_amr.urdf", low_amr_opts)

# =================================================================
# 4. 직사각형 환경 생성 및 외벽/기둥 배치
# =================================================================
env = gym.create_env(sim, gymapi.Vec3(-room_width/2, -room_length/2, 0), gymapi.Vec3(room_width/2, room_length/2, wall_height), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

# ★ 가로/세로 절반 길이 계산
half_w = room_width / 2
half_l = room_length / 2

w_back = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_w, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_w, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_l, wall_height/2)), "wall_right", 0, 0)
w_left = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_l, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
	gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

p_offset_w = half_w - 0.25
p_offset_l = half_l - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset_w, p_offset_w]:
	for y in [-p_offset_l, p_offset_l]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)),f"corner_pillar_{x}_{y}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

pillar_side_positions = np.arange(-2.5, 3.5, 2.5)
for y_pos in pillar_side_positions:
	for x_pos in [-p_offset_w, p_offset_w]:
		p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height/2)),f"side_pillar_{x_pos}_{y_pos}", 0, 0)
		gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# =================================================================
# 🏭 [레이아웃 재설계] 양쪽 벽면 밀착 정렬 & 중앙 로봇 고속도로 확보
# =================================================================
# 좌측(X=-3.5)과 우측(X=3.5) 벽을 따라 랙을 6개씩 일렬로 배치합니다.
# 저상형 AMR의 타겟 좌표인 [3.5, 0.77] 위치에는 랙이 정확히 위치하도록 좌표를 맞췄습니다.
shelf_positions = [
	# 좌측 벽면 (X=-2.3)
	[-2.3, 3.49], [-2.3, 2.13], [-2.3, 0.77], [-2.3, -0.59],
	# 뒤쪽 벽면 (Y=5.0)
	[-2.3, 5.0], [-0.5, 5.0], [1.0, 5.0], [2.5, 5.0],
	# 우측 벽면 (X=2.3)
	[2.3, 3.49], [2.3, 2.13], [2.3, 0.77], [2.3, -0.59], [2.3, -1.95], [2.3, -3.31], [2.3, -4.76]
]

cardboard_color = gymapi.Vec3(0.76, 0.60, 0.42)

for idx, pos in enumerate(shelf_positions):
	pose = gymapi.Transform(p=gymapi.Vec3(pos[0], pos[1], 0.06))
	s_handle = gym.create_actor(env, cargo_shelf_asset, pose, f"shelf_{idx}", -1, 0)

	s_props = gym.get_actor_rigid_shape_properties(env, s_handle)
	for s in s_props:
		s.filter = 2
		s.friction = 5.0
	gym.set_actor_rigid_shape_properties(env, s_handle, s_props)

	# 5개의 상단 선반 각각에 4x4 그리드(16개) 박스를 빈틈없이 적재 (총 960여 개)
	for z_level in [1.15, 2.05, 2.95, 3.85, 4.75]:
		for dx in [-0.38, -0.13, 0.13, 0.38]:
			for dy in [-0.38, -0.13, 0.13, 0.38]:
				rand_x = dx + np.random.uniform(-0.01, 0.01)
				rand_y = dy + np.random.uniform(-0.01, 0.01)
				b_pose = gymapi.Transform(p=gymapi.Vec3(pos[0] + rand_x, pos[1] + rand_y, z_level + 0.15))

				b_handle = gym.create_actor(env, box_asset, b_pose, f"cargo_box_{idx}_{z_level}_{dx}_{dy}", -1, 0)
				gym.set_rigid_body_color(env, b_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cardboard_color)

				b_props = gym.get_actor_rigid_shape_properties(env, b_handle)
				for b in b_props: b.friction = 3.0
				gym.set_actor_rigid_shape_properties(env, b_handle, b_props)

# =================================================================
# generate a* map
# =================================================================
from map_generator import AStarMapGenerator
from datetime import datetime

# 하드코딩 없이 기존 메인 파일 상단의 변수들을 그대로 주입
map_resolution = 0.05       # 1픽셀 = 5cm 해상도 설정
shelf_width_m = 1.4         # urdf 선반 실제 가로 규격(m) 선반의 urdf data가 달라지면 그에 맞게 바꿔야 함
shelf_length_m = 1.4        # urdf 선반 실제 세로 규격(m) 선반의 urdf data가 달라지면 그에 맞게 바꿔야 함

# 1. 제너레이터 초기화
generator = AStarMapGenerator(
    room_width=room_width, 
    room_length=room_length, 
    resolution=map_resolution, 
    wall_thickness=wall_thickness
)

# 2. 정적 장애물 순차적 마킹 작업 수행
generator.mark_walls()  # 외벽 자동 마킹

# 선반 레이아웃 좌표 리스트 전달
generator.mark_rectangular_obstacles(shelf_positions, shelf_width_m, shelf_length_m) #####

# 코너 및 사이드 기둥 위치 조합 생성 후 마킹
pillar_positions = []
for x in [-p_offset_w, p_offset_w]:
    for y in [-p_offset_l, p_offset_l]:
        pillar_positions.append([x, y])
for y_pos in pillar_side_positions:
    for x_pos in [-p_offset_w, p_offset_w]:
        pillar_positions.append([x_pos, y_pos])

generator.mark_rectangular_obstacles(pillar_positions, pillar_x_size, pillar_y_size)

# 3. 맵 파일 물리적 내보내기 (기존 경로 자동 매핑)

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

dynamic_npy_name = f"map_{current_time_str}.npy"
dynamic_yaml_name = f"map_{current_time_str}.yaml"
dynamic_png_name = f"map_{current_time_str}.png"

map_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map1")
generator.save_map(map_dir, dynamic_npy_name, dynamic_yaml_name, dynamic_png_name)

# =================================================================
# 동적 생성된 맵 기반 마우스 선택 인터페이스 가동
# =================================================================
import matplotlib.pyplot as plt

def get_target_by_mouse_click(png_path, room_width, room_length, robot_poses=None, current_time_str=""):
    """
    저장된 고유 맵 이미지를 로드하고, 로봇들의 현재 위치를 180도 반전 매핑하여 마킹한 뒤 
    마우스 클릭한 픽셀 좌표를 다시 180도 역산하여 실제 월드 목적지 좌표(m)를 추출합니다.
    """
    print(f"\n[인터페이스] 팝업 창에서 목적지를 마우스로 클릭해 주세요.")
    print(f" -> 로드된 최신 맵 이미지: {os.path.basename(png_path)}")
    
    img = plt.imread(png_path)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, origin='lower')
    ax.set_title(f"Set Goal Target ({current_time_str})", fontsize=12, fontweight='bold')
    ax.axis('on')
    
    img_height, img_width = img.shape[0], img.shape[1]
    
    # 🚀 [180도 보정] 로봇들의 현재 위치 변환 (1.0 - ratio 적용)
    if robot_poses is not None:
        # Carter (바구니 AMR) 위치 마킹
        if 'carter' in robot_poses:
            cx, cy = robot_poses['carter']
            # 원래 비율을 구한 뒤 1.0에서 빼서 축을 반대로 뒤집음
            ratio_cx = 1.0 - ((cx + room_width / 2.0) / room_width)
            ratio_cy = 1.0 - ((cy + room_length / 2.0) / room_length)
            
            pixel_cx = ratio_cx * img_width
            pixel_cy = ratio_cy * img_height
            ax.scatter(pixel_cx, pixel_cy, c='blue', s=150, marker='o', edgecolors='white', label='Carter AMR (Start)')
            ax.text(pixel_cx + 10, pixel_cy, 'Carter', color='blue', fontsize=10, fontweight='bold')
            
        # LowProfileAMR (저상형 AMR) 위치 마킹
        if 'low_amr' in robot_poses:
            lx, ly = robot_poses['low_amr']
            ratio_lx = 1.0 - ((lx + room_width / 2.0) / room_width)
            ratio_ly = 1.0 - ((ly + room_length / 2.0) / room_length)
            
            pixel_lx = ratio_lx * img_width
            pixel_ly = ratio_ly * img_height
            ax.scatter(pixel_lx, pixel_ly, c='red', s=150, marker='s', edgecolors='white', label='Low AMR (Start)')
            ax.text(pixel_lx + 10, pixel_ly, 'Low AMR', color='red', fontsize=10, fontweight='bold')
            
        ax.legend(loc='upper right')

    clicked_coords = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            pixel_x, pixel_y = event.xdata, event.ydata
            
            # 🚀 [180도 보정] 클릭 픽셀 위치 비율을 역산할 때도 반전(1.0 - ratio) 처리
            ratio_x_flipped = pixel_x / img_width
            ratio_y_flipped = pixel_y / img_height
            
            ratio_x = 1.0 - ratio_x_flipped
            ratio_y = 1.0 - ratio_y_flipped
            
            # 반전 복원된 비율로 실제 월드(m) 좌표 정밀 계산
            world_x = ratio_x * room_width - (room_width / 2.0)
            world_y = ratio_y * room_length - (room_length / 2.0)
            
            clicked_coords.append((world_x, world_y))
            print(f"🎯 180도 축 보정 완료! 월드(m): [{world_x:.2f}, {world_y:.2f}]")
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    if not clicked_coords:
        return np.array([0.0, 0.0])
    return np.array(clicked_coords[0])

# =================================================================
# 🛠️ 로봇 인스턴스 스폰 (중앙 하이웨이 배치)
# =================================================================
small_box_asset = gym.create_box(sim, 0.07, 0.07, 0.07, gymapi.AssetOptions())
small_box_handle = gym.create_actor(env, small_box_asset, gymapi.Transform(p=gymapi.Vec3(-0.3, 0.0, 0.1)), "small_box", -1, 0)

carter_handle = gym.create_actor(env, carter_asset, gymapi.Transform(p=gymapi.Vec3(0, 2.7, 0.30)), "carter", -1, 0)
tray_handle = gym.create_actor(env, tray_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.7, 0.75)), "tray", -1, 0)

low_amr_handle = gym.create_actor(env, low_amr_asset, gymapi.Transform(p=gymapi.Vec3(0.5, 0.77, 0.20)), "low_amr", -1, 0)

low_amr_props = gym.get_actor_rigid_shape_properties(env, low_amr_handle)
for s in low_amr_props:
	s.friction = 10.0
	s.rolling_friction = 5.0
gym.set_actor_rigid_shape_properties(env, low_amr_handle, low_amr_props)

# 목적지 시각화용 투명한 구(Sphere) 에셋 정의
marker_opts = gymapi.AssetOptions()
marker_opts.fix_base_link = True
marker_opts.disable_gravity = True

# 반지름 0.15m 크기의 마커 에셋 생성
marker_asset = gym.create_sphere(sim, 0.15, marker_opts)

# 🚀 초기 위치는 일단 보이지 않게 땅속이나 원점에 스폰 (클릭 후 좌표 이동 예정)
marker_carter_h = gym.create_actor(env, marker_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, -1.0)), "marker_carter", -1, 0)
marker_low_h = gym.create_actor(env, marker_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, -1.0)), "marker_low_amr", -1, 0)

# 마커 색상 지정 (Carter 목적지 = 파란색, Low AMR 목적지 = 빨간색)
gym.set_rigid_body_color(env, marker_carter_h, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.3, 1.0))
gym.set_rigid_body_color(env, marker_low_h, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.1, 0.1))

################################

# --- A* 플래너 및 컨트롤러 연동 ---
npy_path = os.path.join(map_dir, dynamic_npy_name)
yaml_path = os.path.join(map_dir, dynamic_yaml_name)

carter_planner = AStarPlanner(map_npy_path=npy_path, map_yaml_path=yaml_path, robot_radius=0.6)
amr_ctrl = CarterAMR(gym, env, carter_handle, tray_handle, planner=carter_planner)

low_planner = AStarPlanner(map_npy_path=npy_path, map_yaml_path=yaml_path, robot_radius=0.25)
low_amr_ctrl = LowProfileAMR(gym, env, low_amr_handle, planner=low_planner)

# 사용자 좌표 입력 수신
carter_init_state = gym.get_actor_rigid_body_states(env, carter_handle, gymapi.STATE_ALL)
low_amr_init_state = gym.get_actor_rigid_body_states(env, low_amr_handle, gymapi.STATE_ALL)

current_robot_poses = {
    'carter': [
        carter_init_state['pose']['p'][0]['x'], 
        carter_init_state['pose']['p'][0]['y']
    ],
    'low_amr': [
        low_amr_init_state['pose']['p'][0]['x'], 
        low_amr_init_state['pose']['p'][0]['y']
    ]
}

target_map_png_path = os.path.join(map_dir, dynamic_png_name)

print("\n>>> [1단계] 바구니 AMR 목적지 선택 (파란 원: 로봇 현재 위치)")
target_pos = get_target_by_mouse_click(
    target_map_png_path, room_width, room_length, robot_poses=current_robot_poses
)

print("\n>>> [2단계] 저상형 AMR 배달 목적지 선택 (빨간 네모: 로봇 현재 위치)")
low_delivery_coord = get_target_by_mouse_click(
    target_map_png_path, room_width, room_length, robot_poses=current_robot_poses
)

print(f"\n[시스템] 마우스로 선택한 좌표를 Isaac Gym 월드 마커로 동기화합니다.")

# 1. Carter AMR 목적지 마커 상태 추출 및 텔레포트
# 복수형 딕셔너리 구조 배열을 가져옵니다. (기존 메인 코드 내부의 방식을 그대로 차용)
carter_marker_states = gym.get_actor_rigid_body_states(env, marker_carter_h, gymapi.STATE_ALL)

# 0번 인덱스(Root Link)의 포지션 XY를 마우스 클릭 좌표로 가로챕니다.
carter_marker_states['pose']['p'][0]['x'] = target_pos[0]
carter_marker_states['pose']['p'][0]['y'] = target_pos[1]
carter_marker_states['pose']['p'][0]['z'] = 1.0  # 지면에서 살짝 뜨게 마킹

# 속도와 각속도 성분도 초기화하여 물리적 관성 낙하를 방지합니다.
carter_marker_states['vel']['linear'][0] = (0.0, 0.0, 0.0)
carter_marker_states['vel']['angular'][0] = (0.0, 0.0, 0.0)

# 🚀 공식 복수형 API로 시뮬레이터 공간에 세팅
gym.set_actor_rigid_body_states(env, marker_carter_h, carter_marker_states, gymapi.STATE_ALL)


# 2. Low Profile AMR 배달 목적지 마커 상태 추출 및 텔레포트
low_marker_states = gym.get_actor_rigid_body_states(env, marker_low_h, gymapi.STATE_ALL)

low_marker_states['pose']['p'][0]['x'] = low_delivery_coord[0]
low_marker_states['pose']['p'][0]['y'] = low_delivery_coord[1]
low_marker_states['pose']['p'][0]['z'] = 1.0

low_marker_states['vel']['linear'][0] = (0.0, 0.0, 0.0)
low_marker_states['vel']['angular'][0] = (0.0, 0.0, 0.0)

# 🚀 공식 복수형 API로 시뮬레이터 공간에 세팅
gym.set_actor_rigid_body_states(env, marker_low_h, low_marker_states, gymapi.STATE_ALL)

###################

initial_c_states = gym.get_actor_rigid_body_states(env, carter_handle, gymapi.STATE_ALL)
amr_ctrl.set_target((initial_c_states['pose']['p'][0]['x'], initial_c_states['pose']['p'][0]['y']), target_pos)

start_pos_coord = np.array([low_amr_init_state['pose']['p'][0]['x'], low_amr_init_state['pose']['p'][0]['y']])
# target_pos_coord = np.array([3.5, 0.77])

low_amr_ctrl.set_lift("DOWN")
low_amr_ctrl.current_path = low_planner.plan_path(
    start_pos_coord, 
    low_delivery_coord, 
    robot_name="low_amr", 
    save_debug_img=True
)
low_amr_ctrl.current_wp_idx = 0

low_amr_ctrl.viz_start_pos = start_pos_coord
low_amr_ctrl.viz_goal_pos = low_delivery_coord

low_amr_state = "GO_TO_TARGET_COORD"
low_lift_timer = 0

c_props = gym.get_actor_dof_properties(env, carter_handle)
c_props["driveMode"].fill(gymapi.DOF_MODE_VEL)
gym.set_actor_dof_properties(env, carter_handle, c_props)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# 바뀐 레이아웃(중앙 고속도로)이 한눈에 보이도록 카메라를 정면 끝단에 배치
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-3.5, -4.5, 12.0), gymapi.Vec3(0.0, 0.0, 0.0))

frame_count = 0

while not gym.query_viewer_has_closed(viewer):

	curr_p, curr_r = amr_ctrl.sync_tray()
	amr_ctrl.drive_to_target(curr_p, curr_r, True)
	
	low_p, low_r = low_amr_ctrl.get_current_pose()
	low_amr_ctrl.drive_to_target(low_p, low_r)
	
	gym.simulate(sim)
	gym.fetch_results(sim, True)
	gym.step_graphics(sim)
	gym.draw_viewer(viewer, sim, True)
	
	gym.sync_frame_time(sim)
	frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)