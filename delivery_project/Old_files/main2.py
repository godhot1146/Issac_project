import os
from isaacgym import gymapi
from isaacgym import gymutil

# 1. Isaac Gym 초기화 및 아규먼트 파싱 (기본 GUI 창 띄우기용)
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Warehouse Asset Viewer")

# 2. 시뮬레이션 기본 설정 (업방향 Z축 설정 등)
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# PhysX 엔진 설정 (가장 일반적인 세팅)
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.02
sim_params.physx.rest_offset = 0.001

# 그래픽 파라미터 아규먼트 반영하여 Sim 생성
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# 3. 그라운드 플레인(바닥 지면) 생성
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) # Z축이 위를 향함
plane_params.distance = 0.0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0

gym.add_ground(sim, plane_params)

# 4. 에셋 옵션 세팅 및 로드
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True           # 공장 랙이 바닥에 단단히 고정되도록 True
asset_options.use_mesh_materials = True      # 텍스처 데이터 이미지 매핑 허용
asset_options.convex_decomposition_from_submeshes = True # 메쉬 쪼개서 충돌 정밀 계산

# 에셋 경로 정의 (사용자 환경의 절대경로 지정)
asset_root = "/home/hprobot/isaacgym/assets/warehouse"
asset_file = "high_rack.urdf"

print(f"Loading asset from {asset_root}/{asset_file}...")
rack_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 5. 환경(Env) 생성 및 에셋 배치
# Isaac Gym은 하나의 큰 공간(Sim) 안에 여러 작은 구역(Env)을 나눕니다. 여기서는 1개만 만듭니다.
num_envs = 1
env_spacing = 5.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

env_ptr = gym.create_env(sim, env_lower, env_upper, 1)

# 랙 배치 포즈 세팅 (원점 x=0, y=0, z=0 바닥에 안착)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0, 0, 0, 1)

# 에셋을 하나의 액터(Actor) 객체로 환경에 등록
rack_actor = gym.create_actor(env_ptr, rack_asset, pose, "high_rack", 0, 0)

# 6. 시각화용 뷰어(Viewer) 카메라 생성
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# 카메라가 원점에 있는 랙을 바라보도록 위치 조정 (x=3m, y=3m, z=3m 지점에서 원점을 바라봄)
cam_pos = gymapi.Vec3(3.0, 3.0, 3.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, env_ptr, cam_pos, cam_target)

# 7. 시뮬레이션 메인 루프 돌리기
print("Simulation running. Close the viewer window to exit.")
while not gym.query_viewer_has_closed(viewer):
    # 물리 스텝 연산
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 뷰어 그래픽스 갱신
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # 동기화 대기
    gym.sync_frame_time(sim)

# 종료 후 메모리 정리
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)