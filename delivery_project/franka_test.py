import os
import numpy as np
from isaacgym import gymapi, gymutil

# 1. 시뮬레이션 및 물리 엔진 초기화
gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="Franka Pick, Place and Return")

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.use_gpu = True
sim_params.physx.num_position_iterations = 12 
sim_params.physx.num_velocity_iterations = 1

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# 2. 바닥 생성
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 3. 에셋 로드
asset_root = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/isaac_assets")
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

box_size = 0.07
box_asset = gym.create_box(sim, box_size, box_size, box_size, gymapi.AssetOptions())

# 4. 환경 및 액터 생성
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

# 로봇 생성
franka_handle = gym.create_actor(env, franka_asset, gymapi.Transform(), "franka", 0, 1)

# 상자 생성 (하얀색)
box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(0.67, 0.0, box_size/2 + 0.01)
box_handle = gym.create_actor(env, box_asset, box_pose, "box", 0, 0)
gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 1))

# 5. 물리 및 제어 설정
dof_props = gym.get_actor_dof_properties(env, franka_handle)
dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
dof_props["stiffness"].fill(800.0)
dof_props["damping"].fill(40.0)
gym.set_actor_dof_properties(env, franka_handle, dof_props)

shape_props = gym.get_actor_rigid_shape_properties(env, franka_handle)
for s in range(len(shape_props)):
    shape_props[s].friction = 5.0
gym.set_actor_rigid_shape_properties(env, franka_handle, shape_props)

# 6. 정밀 동작 시퀀스 정의 (각도 튜닝)
# [Joint 0, 1, 2, 3, 4, 5, 6, Gripper_L, Gripper_R]
ready_pose     = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]
# pick_pose      = [0, 0.63, 0, -1.75, 0, 2.12, 0.785, 0.04, 0.04] 
pick_pose = [0, 0.8, 0, -1.75, 0, 2.50, 0.90, 0.04, 0.04]
close_pose     = [0, 0.8, 0, -1.75, 0, 2.50, 0.90, 0.0, 0.0]   
lift_pose      = [0, -0.5, 0, -2.0, 0, 1.50, 0.90, 0.0, 0.0]    

# 우측 90도 회전 및 내려놓기 시퀀스
rotate_rt_pose = [-1.571, -0.5, 0, -2.0, 0, 1.5, 0.785, 0.0, 0.0]   # 물건 든 채 오른쪽 회전
place_pose     = [-1.571, 0.63, 0, -1.75, 0, 2.12, 0.785, 0.0, 0.0]  # 내려놓기 하강
release_pose   = [-1.571, 0.63, 0, -1.75, 0, 2.12, 0.785, 0.04, 0.04] # 그리퍼 개방
lift_up_pose   = [-1.571, -0.5, 0, -2.0, 0, 1.5, 0.785, 0.04, 0.04]  # 물건 두고 위로 상승
return_home    = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04] # 원위치 복귀

# 초기 자세 적용
dof_states = gym.get_actor_dof_states(env, franka_handle, gymapi.STATE_NONE)
for i in range(len(ready_pose)):
    dof_states['pos'][i] = ready_pose[i]
gym.set_actor_dof_states(env, franka_handle, dof_states, gymapi.STATE_POS)

# 7. 실행 루프
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.5, 0, 0.2))

frame_count = 0
print("전체 시퀀스 시작: Pick -> Rotate -> Place -> Return")

while not gym.query_viewer_has_closed(viewer):
    # 시점별 동작 스케줄링 (각 단계당 120프레임 = 약 2초)
    if frame_count < 120:
        targets = ready_pose
    elif frame_count < 240:
        targets = pick_pose
    elif frame_count < 360:
        targets = close_pose
    elif frame_count < 480:
        targets = lift_pose
    elif frame_count < 600:
        targets = rotate_rt_pose # 1. 오른쪽 90도 회전
    elif frame_count < 720:
        targets = place_pose     # 2. 물건 냅두기 위해 하강
    elif frame_count < 840:
        targets = release_pose   # 3. 그리퍼 열기
    elif frame_count < 960:
        targets = lift_up_pose   # 4. 다시 위로 올라오기
    elif frame_count < 1080:
        targets = return_home    # 5. 왼쪽으로 90도 돌아 원상태 복귀
    else:
        targets = ready_pose     # 대기

    gym.set_actor_dof_position_targets(env, franka_handle, np.array(targets, dtype=np.float32))

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
