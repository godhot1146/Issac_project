import os
import math
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

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

gym.set_light_parameters(sim, 0, gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1.0, 0.0, -1.0))

# 2. 바닥 평면 생성
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = -0.05
gym.add_ground(sim, plane_params)

# =================================================================
# 3. [25평 공장 레이아웃 변형] 에셋 규격 및 가상 맵 정의
#    25평 규모(9.1m x 9.1m) 구조를 유지하되 내측 적재 랙은 모두 제거합니다.
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

# 25평 정사각형 공장 사양 (9.1m * 9.1m)
room_size = 9.1          
wall_height = 10.0        # 공장 층고 (10m)
wall_thickness = 0.2     # 외벽 두께 (20cm)

floor_asset = gym.create_box(sim, room_size, room_size, 0.05, env_opts)
wall_x = gym.create_box(sim, wall_thickness, room_size, wall_height, env_opts)
wall_y = gym.create_box(sim, room_size, wall_thickness, wall_height, env_opts)
pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height, env_opts) # 코너 H빔 기둥

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_opts.disable_gravity = False
tote_opts.armature = 0.01
tote_opts.override_inertia = True
tote_opts.override_com = True
tote_asset = gym.load_asset(sim, asset_root, "urdf/tote/whole/test.urdf", tote_opts)

# =================================================================
# 4. 25평 오픈 팩토리 환경 생성 및 외벽/기둥 배치 (Actors)
# =================================================================
env = gym.create_env(sim, gymapi.Vec3(-room_size/2, -room_size/2, 0), gymapi.Vec3(room_size/2, room_size/2, room_size), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

# 25평 외곽 경계벽 스폰
half_r = room_size / 2
w_back  = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_r, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_r, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_r, wall_height/2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_r, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
    gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

# 공장 프레임 유지를 위한 코너 사각 기둥 배치
p_offset = half_r - 0.25
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset, p_offset]:
    for y in [-p_offset, p_offset]:
        p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)),f"corner_pillar_{x}_{y}", 0, 0)
        gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# 외벽 보강 지지 기둥 정렬
pillar_side_positions = np.arange(-2.5, 3.5, 2.5)
for y_pos in pillar_side_positions:
    for x_pos in [-p_offset, p_offset]:
        p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x_pos, y_pos, wall_height/2)),f"side_pillar_{x_pos}_{y_pos}", 0, 0)
        gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# --- 로봇 인스턴스 스폰 ---
tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 1.5)), "tote_asset", -1, 0)

# 1. get_actor_joint_properties 대신 get_actor_dof_properties를 사용합니다.
dof_props = gym.get_actor_dof_properties(env, tote_handle)
dof_dict = gym.get_actor_dof_dict(env, tote_handle)

# 1. 모든 관절 속성 초기화
for i in range(len(dof_props)):
    # 상부 관절들이 속도 0으로 뻗대는 것을 막기 위해 힘을 완전히 빼버립니다. (중력 흐름에 맡김)
    dof_props[i]['driveMode'] = gymapi.DOF_MODE_NONE 
    dof_props[i]['stiffness'] = 0.0
    dof_props[i]['damping'] = 0.0
    dof_props[i]['effort'] = 0.0

# 2. 오직 바퀴 2개만 속도 제어(VEL) 모드로 명확히 지정
wheel_joints = ['left_wheel_joint', 'right_wheel_joint']
for joint_name in wheel_joints:
    if joint_name in dof_dict:
        idx = dof_dict[joint_name]
        dof_props[idx]['driveMode'] = gymapi.DOF_MODE_VEL  # 바퀴만 VEL 모드 활성화
        dof_props[idx]['stiffness'] = 0.0
        dof_props[idx]['damping'] = 1.0         # 안정적인 구동 마찰력
        dof_props[idx]['effort'] = 500000.0      # 강력한 가속 토크 보장
        # [주의] dof_props[idx]['hasLimits'] = False 라인은 제거 상태 유지 (Read-only 에러 방지)

# 수정한 속성을 액터에 다시 적용
gym.set_actor_dof_properties(env, tote_handle, dof_props)

# 3. 지면 마찰력 정돈
tote_shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
for shape in tote_shape_props:
    shape.friction = 3.0          # 접지력 확보
    shape.rolling_friction = 0.01
    shape.restitution = 0.0
gym.set_actor_rigid_shape_properties(env, tote_handle, tote_shape_props)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

frame_count = 0

left_wheel_idx = dof_dict['left_wheel_joint']
right_wheel_idx = dof_dict['right_wheel_joint']
lift_z_idx = dof_dict['lift_z_joint']
lift_rot_idx = dof_dict['lift_rotation_joint']
fork_ext_idx = dof_dict['fork_extension_joint']
claw_idx = dof_dict['left_claw_joint']      # 왼쪽 문 역할
claw2_idx = dof_dict['right_claw_joint']    # 오른쪽 문 역할


TARGET_ROTATION = math.pi / 2

sphere_origin = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(1, 1, 1)) # 원점: 흰색
sphere_x      = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(1, 0, 0)) # X축: 빨간색 (Red)
sphere_y      = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(0, 1, 0)) # Y축: 초록색 (Green)
sphere_z      = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=8, num_lons=8, color=(0, 0, 1)) # Z축: 파란색 (Blue)

# viewer = gym.create_viewer(...) 위아래 적절한 위치에 추가
mast_rb_idx = gym.find_actor_rigid_body_index(env, tote_handle, "mast_link", gymapi.DOMAIN_ACTOR)
print(f"-> Detected mast_link RigidBody Index: {mast_rb_idx}")

# 시각화에 사용할 구체 형상 정의 (반지름 8cm)
sphere_mast_origin = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=12, num_lons=12, color=(1, 1, 0)) # 원점: 노란색
sphere_mast_com    = gymutil.WireframeSphereGeometry(radius=0.1, num_lats=12, num_lons=12, color=(1, 0, 1)) # 무게중심: 자홍색(Magenta)

# 1. 에셋의 전체 강체(Rigid Body) 관성 속성들을 가져옵니다.
mast_link_properties = gym.get_actor_rigid_body_properties(env, tote_handle)

# 2. .com 속성 자체가 이미 gymapi.Vec3 (x, y, z 좌표) 객체입니다.
local_com_offset = mast_link_properties[mast_rb_idx].com

# =================================================================
# [수정] 각 부위별로 원점(org)과 무게중심(com)의 크기를 계층화
# =================================================================
track_links = {
    "left_wheel_link":      {"idx": -1, "color": (1, 0, 0),    "radius_org": 0.1, "radius_com": 0.1}, # 바퀴 (빨강)
    "right_wheel_link":     {"idx": -1, "color": (1, 0, 0),    "radius_org": 0.1, "radius_com": 0.1},    
    "lift_shuttle_link":    {"idx": -1, "color": (0, 1, 1),    "radius_org": 0.05, "radius_com": 0.03}, # 리프트 (청록)
    "shuttle_rotator_link": {"idx": -1, "color": (0, 0, 1),    "radius_org": 0.05, "radius_com": 0.03}, # 회전부 (파랑)
    "fork_arm_link":        {"idx": -1, "color": (1, 0, 1),    "radius_org": 0.05, "radius_com": 0.03}, # 포크암 (자홍)
    "left_claw_link":       {"idx": -1, "color": (1, 0.5, 0),  "radius_org": 0.05, "radius_com": 0.03}, # 클로 (주황)
    "right_claw_link":      {"idx": -1, "color": (1, 0.5, 0),  "radius_org": 0.05, "radius_com": 0.03}
}

print(dof_dict)

for name, idx in dof_dict.items():
    print(
        idx,
        name,
        dof_props[idx]['driveMode'],
        dof_props[idx]['effort']
    )

print(
    "num_dofs =",
    gym.get_asset_dof_count(tote_asset)
)

print(
    "num_bodies =",
    gym.get_asset_rigid_body_count(tote_asset)
)

rb_names = gym.get_actor_rigid_body_names(
    env,
    tote_handle
)

print(rb_names)

shape_count = gym.get_actor_rigid_shape_count(
    env,
    tote_handle
)

print(shape_count)

shape_props = gym.get_actor_rigid_shape_properties(
    env,
    tote_handle
)

print(len(shape_props))

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    num_dof = gym.get_actor_dof_count(env, tote_handle)
    vel_targets = np.zeros(num_dof, dtype=np.float32)

    vel_targets[left_wheel_idx] = 5.0
    vel_targets[right_wheel_idx] = -5.0
    
    gym.set_actor_dof_velocity_targets(env, tote_handle, vel_targets)

    body_idx = gym.find_actor_rigid_body_index(
        env,
        tote_handle,
        "base_link",
        gymapi.DOMAIN_ACTOR
    )

    gym.apply_body_force_at_pos(
        env,
        body_idx,
        gymapi.Vec3(1000,0,0),
        gymapi.Vec3(0,0,0),
        gymapi.ENV_SPACE
    )
    
    # -----------------------------------------------------------------

    # 실시간 속도 상태 출력
    dof_states = gym.get_actor_dof_states(env, tote_handle, gymapi.STATE_ALL)
    if frame_count % 60 == 0:
        print(f"Frame {frame_count} | L_Vel: {dof_states['vel'][left_wheel_idx]:.4f} / R_Vel: {dof_states['vel'][right_wheel_idx]:.4f}")
    
    rb_states = gym.get_actor_rigid_body_states(
        env,
        tote_handle,
        gymapi.STATE_ALL
    )

    base_idx = gym.find_actor_rigid_body_index(
        env,
        tote_handle,
        "base_link",
        gymapi.DOMAIN_ACTOR
    )
    if frame_count % 60 == 0:
        print(
            rb_states["pose"]["p"][base_idx]
        )

    dof_states = gym.get_actor_dof_states(
        env,
        tote_handle,
        gymapi.STATE_ALL
    )
    if frame_count % 60 == 0:
        print(
            dof_states['pos'][left_wheel_idx],
            dof_states['pos'][right_wheel_idx]
        )
    
    ####### 디버깅 구체 그리기 ########

    # 1. 매 프레임 이전 디버그 라인 초기화
    gym.clear_lines(viewer)

    # 2. 축 가이드 구체 그리기 (현재 에셋 스폰 높이인 z=1.0 부근을 기준으로 설정)
    base_z = 1.0
    
    # [원점] 흰색 구체 (0.0, 0.0, 1.0)
    t_origin = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, base_z))
    gymutil.draw_lines(sphere_origin, gym, viewer, env, t_origin)
    
    # [X축 방향] 빨간색 구체 2개 (Forward 확인용)
    for i in [0.5, 1.0]:
        t_x = gymapi.Transform(p=gymapi.Vec3(i, 0.0, base_z))
        gymutil.draw_lines(sphere_x, gym, viewer, env, t_x)
        
    # [Y축 방향] 초록색 구체 2개 (Right/Left 확인용)
    for i in [0.5, 1.0]:
        t_y = gymapi.Transform(p=gymapi.Vec3(0.0, i, base_z))
        gymutil.draw_lines(sphere_y, gym, viewer, env, t_y)
        
    # [Z축 방향] 파란색 구체 2개 (Up 확인용)
    for i in [0.5, 1.0]:
        t_z = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, base_z + i))
        gymutil.draw_lines(sphere_z, gym, viewer, env, t_z)

    # ==========================================
    # [추가] mast_link 원점 및 무게중심(CoM) 시각화
    # ==========================================
    if mast_rb_idx != -1:
        rb_states = gym.get_actor_rigid_body_states(env, tote_handle, gymapi.STATE_ALL)
        mast_pose = rb_states['pose'][mast_rb_idx]
        
        p_mast = gymapi.Vec3(mast_pose['p'][0], mast_pose['p'][1], mast_pose['p'][2])
        r_mast = gymapi.Quat(mast_pose['r'][0], mast_pose['r'][1], mast_pose['r'][2], mast_pose['r'][3])
        
        # A. 원점 시각화 (노란색)
        t_mast_origin = gymapi.Transform(p=p_mast, r=r_mast)
        gymutil.draw_lines(sphere_mast_origin, gym, viewer, env, t_mast_origin)
        
        # B. 자동으로 받아온 local_com_offset을 회전시켜 전역 CoM 좌표 계산 (자홍색)
        global_com_offset = r_mast.rotate(local_com_offset)
        p_mast_com = p_mast + global_com_offset
        
        t_mast_com = gymapi.Transform(p=p_mast_com, r=r_mast)
        gymutil.draw_lines(sphere_mast_com, gym, viewer, env, t_mast_com)

    # 2. 로봇의 실시간 전역 상태 행렬(Pose) 가져오기
    rb_states = gym.get_actor_rigid_body_states(env, tote_handle, gymapi.STATE_ALL)

    # 3. 등록된 모든 구동 부위 링크들을 순회하며 시각화
    for link_name, info in track_links.items():
        rb_idx = info["idx"]
        if rb_idx == -1:
            continue
            
        link_pose = rb_states['pose'][rb_idx]
        p_link = gymapi.Vec3(link_pose['p'][0], link_pose['p'][1], link_pose['p'][2])
        r_link = gymapi.Quat(link_pose['r'][0], link_pose['r'][1], link_pose['r'][2], link_pose['r'][3])
        
        # --- A. 링크 '원점(Origin)' 동적 생성 및 그리기 ---
        # 원점은 흰색 테두리로, 지정된 고유 크기(radius_org)로 생성됩니다.
        sphere_origin_geom = gymutil.WireframeSphereGeometry(
            radius=info["radius_org"], num_lats=8, num_lons=8, color=(1, 1, 1)
        )
        t_origin = gymapi.Transform(p=p_link, r=r_link)
        gymutil.draw_lines(sphere_origin_geom, gym, viewer, env, t_origin)
        
        # --- B. 링크 '무게중심(CoM)' 동적 생성 및 그리기 ---
        global_com_offset = r_link.rotate(info["local_com"])
        p_com = p_link + global_com_offset
        t_com = gymapi.Transform(p=p_com, r=r_link)
        
        # 무게중심은 고유 컬러와 조금 더 작은 크기(radius_com)로 안쪽에 배치됩니다.
        sphere_com_geom = gymutil.WireframeSphereGeometry(
            radius=info["radius_com"], num_lats=10, num_lons=10, color=info["color"]
        )
        gymutil.draw_lines(sphere_com_geom, gym, viewer, env, t_com)
        
        # --- C. 원점과 무게중심을 잇는 가이드라인 연결 ---
        # 좌표가 완전히 같아 오프셋이 0이더라도 선 그리기가 에러 나지 않으므로 그대로 유지합니다.
        gym.add_lines(viewer, env, 1, 
                      [p_link.x, p_link.y, p_link.z, p_com.x, p_com.y, p_com.z], 
                      [info["color"][0], info["color"][1], info["color"][2]])

    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)