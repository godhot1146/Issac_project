import os
import math
import numpy as np
from isaacgym import gymapi, gymutil
from vacuum_controller_v1 import FrankaController
from low_amr_controller_v1 import LowAMR
from lifting_amr_controller_v1 import LiftingAMR

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
# 3. [10m x 8.5m 공장 레이아웃 변형] 에셋 규격 및 가상 맵 정의
# =================================================================
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
env_opts = gymapi.AssetOptions()
env_opts.fix_base_link = True

# 새로운 직사각형 공장 사양 (10m x 8.5m)
room_x = 10.0         # 가로 크기 (X축)
room_y = 8.5          # 세로 크기 (Y축)
wall_height = 4.0    # 공장 층고 (10m)
wall_thickness = 0.2  # 외벽 두께 (20cm)

# 방 크기에 맞는 바닥 및 벽 에셋 생성
floor_asset = gym.create_box(sim, room_x, room_y, 0.05, env_opts)

# wall_x: Y축 방향으로 길게 뻗은 벽 (X축 경계면에 배치됨, 길이는 room_y)
wall_x = gym.create_box(sim, wall_thickness, room_y, wall_height, env_opts)
# wall_y: X축 방향으로 길게 뻗은 벽 (Y축 경계면에 배치됨, 길이는 room_x)
wall_y = gym.create_box(sim, room_x, wall_thickness, wall_height, env_opts)

pillar_asset = gym.create_box(sim, 0.4, 0.4, wall_height+0.1, env_opts) # 코너 H빔 기둥

# =================================================================
# 4. 오픈 팩토리 환경 생성 및 외벽/기둥 배치 (Actors)
# =================================================================
# 환경 영역(Bounding Box) 정의도 변경된 크기에 맞춤
env = gym.create_env(sim, gymapi.Vec3(-room_x/2, -room_y/2, 0), gymapi.Vec3(room_x/2, room_y/2, wall_height), 1)

floor_h = gym.create_actor(env, floor_asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.03)), "floor", 0, 0)
gym.set_rigid_body_color(env, floor_h, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.65, 0.65, 0.65))

# 직사각형 외곽 경계벽 스폰 위치 계산
half_x = room_x / 2
half_y = room_y / 2

w_back  = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(-half_x, 0, wall_height/2)), "wall_back", 0, 0)
w_front = gym.create_actor(env, wall_x, gymapi.Transform(p=gymapi.Vec3(half_x, 0, wall_height/2)), "wall_front", 0, 0)
w_right = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, -half_y, wall_height/2)), "wall_right", 0, 0)
w_left  = gym.create_actor(env, wall_y, gymapi.Transform(p=gymapi.Vec3(0, half_y, wall_height/2)), "wall_left", 0, 0)

wall_gray = gymapi.Vec3(0.5, 0.5, 0.5)
for w in [w_back, w_front, w_right, w_left]:
    gym.set_rigid_body_color(env, w, 0, gymapi.MESH_VISUAL_AND_COLLISION, wall_gray)

# 공장 프레임 유지를 위한 코너 사각 기둥 배치 (X, Y 축 각각 오프셋 적용)
p_offset_x = half_x - 0.0
p_offset_y = half_y - 0.0
light_gray = gymapi.Vec3(0.7, 0.7, 0.7)

for x in [-p_offset_x, p_offset_x]:
    for y in [-p_offset_y, p_offset_y]:
        p_handle = gym.create_actor(env, pillar_asset, gymapi.Transform(p=gymapi.Vec3(x, y, wall_height/2)), f"corner_pillar_{x}_{y}", 0, 0)
        gym.set_rigid_body_color(env, p_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, light_gray)

# =============================================================================================
# --- 창고 구조물 스폰 --- 
# =============================================================================================

move_rack_opts = gymapi.AssetOptions()
move_rack_opts.fix_base_link = False
move_rack_opts.density = 100.0
move_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/move_rack/move_rack.urdf", move_rack_opts)

lock_rack_opts = gymapi.AssetOptions()
lock_rack_opts.fix_base_link = True
lock_rack_opts.density = 100.0
lock_rack_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/lock_rack/lock_rack.urdf", lock_rack_opts)

box_opts = gymapi.AssetOptions()
box_opts.fix_base_link = False
box_opts.density = 100.0
box_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/box.urdf", box_opts)

tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_opts.density = 100.0
tote_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/tote/tote.urdf", tote_opts)

desk_opts = gymapi.AssetOptions()
desk_opts.fix_base_link = True
desk_opts.density = 100.0
desk_asset = gym.load_asset(sim, asset_root, "urdf/warehouse/desk.urdf", desk_opts)

door_opts = gymapi.AssetOptions()
door_opts.fix_base_link = True
door_opts.density = 100.0
door_asset = gym.load_asset(sim, asset_root, "urdf/door/door1.urdf", door_opts)
door_asset2 = gym.load_asset(sim, asset_root, "urdf/door/door2.urdf", door_opts)

door_handle1 = gym.create_actor(env, door_asset, gymapi.Transform(p=gymapi.Vec3(-4.9, -0.86, 0.0)), f"door_asset", -1, 0)
door_handle2 = gym.create_actor(env, door_asset2, gymapi.Transform(p=gymapi.Vec3(-4.9, 0.86, 0.0)), f"door_asset", -1, 0)

desk_handle = gym.create_actor(env, desk_asset, gymapi.Transform(p=gymapi.Vec3(-2.05, -2.75, 0.5)), f"desk_asset", -1, 0)

# ==============================================================================
# 0. 오프셋 및 위치 변수 설정
# ==============================================================================
MOVE_RACK_X_OFFSET = 0.25
MOVE_RACK_Y_OFFSET = 0.31

LOCK_RACK_X_OFFSET = 0.25
Z_BASE_OFFSET = 0.01

ITEM_FRICTION = 10.0          # 정적/동적 마찰 계수 
ITEM_ROLLING_FRICTION = 5.0   # 구름 마찰 계수 (미끄러짐 억제)
ITEM_RESTITUTION = 0.0        # 반발 탄성 (채터링 및 튀어오름 방지)

# 1. 이동식 선반(Move Rack) 중심 위치 리스트
move_rack_positions = [
    (1.5, 0.7, 0.001),  (2.8, 0.7, 0.001),  (4.1, 0.7, 0.001),
    (1.5, -0.7, 0.001), (2.8, -0.7, 0.001), (4.1, -0.7, 0.001)
]

# 2. 고정식 선반(Lock Rack - 회전 없음) 중심 위치 리스트
lock_rack_positions = [
    (1.4, 3.7, 0.0), (2.75, 3.7, 0.0), (4.1, 3.7, 0.0),
    (1.4, -3.7, 0.0), (2.75, -3.7, 0.0), (4.1, -3.7, 0.0),
    (-1.4, 3.7, 0.0), (-2.75, 3.7, 0.0), (-4.1, 3.7, 0.0),
    (-1.4, 1.5, 0.0), (-2.75, 1.5, 0.0), (-4.1, 1.5, 0.0)
]

# 3. 고정식 선반 (자체 회전 2개) 중심 위치 리스트
rotated_lock_rack_positions = [
    (-4.5, -3.35, 0.0),
    (-4.5, -1.95, 0.0)
]

# ==============================================================================
# [NEW] 특정 물품 제거(제외) 규칙 설정
# ==============================================================================
# 형식 -> 선반의_중심_좌표_튜플: [(층_높이, "위치키방향"), ...]
# 위치 키 가이드: 
#  - Move Rack: "+X-Y", "-X-Y", "+X+Y", "-X+Y"
#  - Lock Rack (회전없음): "-X", "+X"
#  - Lock Rack (회전있음): "-Y", "+Y"
# ------------------------------------------------------------------------------
REMOVE_ITEMS_MAP = {
    # (1.5, 0.7) Move Rack의 1단(0.4층)에서 앞우측(+X-Y) 상자 제거
    (1.5, 0.7, 0.001): [
        (0.4, "+X-Y")
    ],
    
    # (2.75, 3.7) Lock Rack의 2단(1.0층)에서 좌측(-X) 상자 제거
    (2.75, 3.7, 0.0): [
        (1.0, "-X")
    ],
    
    # (-4.5, -3.35) 회전된 Lock Rack의 4단(2.2층)에서 하단(-Y) 상자 제거
    (-4.5, -3.35, 0.0): [
        (2.2, "-Y")
    ],
    
    # 🎯 [FIX 1] Z축 좌표를 0.001로 수정하여 move_rack_positions와 동기화
    (2.8, -0.7, 0.001): [
        (1.0, "+X-Y")
    ],

    # 🎯 [FIX 2] 중복된 키를 하나로 병합하여 -X와 +X 상자가 동시에 지워지도록 수정
    (1.4, -3.7, 0.0): [
        (1.0, "-X"),
        (1.0, "+X")
    ],

    (-2.75, 1.5, 0.0):[
        (1.0, "-X"),
        (1.0, "+X")
    ]
}

def should_skip(rack_pos, height, loc_key):
    """제거 맵에 등록된 물품인지 확인하는 헬퍼 함수"""
    if rack_pos in REMOVE_ITEMS_MAP:
        for skip_height, skip_key in REMOVE_ITEMS_MAP[rack_pos]:
            if abs(skip_height - height) < 0.01 and skip_key == loc_key:
                return True
    return False

# ==============================================================================
# 1. MOVE RACK (이동식 선반) 및 적재물 생성 (선반 정방향 / 박스·토트 90도 회전)
# ==============================================================================
# 박스와 토트에만 적용할 90도 회전 쿼터니언
item_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.570796)

for mx, my, mz in move_rack_positions:
    r_pos = (mx, my, mz)
    # 선반은 회전 없이 정방향으로 생성
    move_rack_handle = gym.create_actor(env, move_rack_asset, gymapi.Transform(p=gymapi.Vec3(mx, my, mz)), "move_rack_asset", -1, 0)
    
    # --------------------------------------------------------------------------
    # 1단 (Z: 0.4) - Box (90도 회전 및 마찰력 적용)
    # --------------------------------------------------------------------------
    if not should_skip(r_pos, 0.4, "+X-Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 0.4 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    if not should_skip(r_pos, 0.4, "-X-Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 0.4 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    if not should_skip(r_pos, 0.4, "+X+Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 0.4 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    if not should_skip(r_pos, 0.4, "-X+Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 0.4 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    # --------------------------------------------------------------------------
    # 2단 (Z: 1.0) - Tote (90도 회전 및 마찰력 적용)
    # --------------------------------------------------------------------------
    if not should_skip(r_pos, 1.0, "+X-Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 1.0 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    if not should_skip(r_pos, 1.0, "-X-Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 1.0 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    if not should_skip(r_pos, 1.0, "+X+Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 1.0 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    if not should_skip(r_pos, 1.0, "-X+Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 1.0 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    # --------------------------------------------------------------------------
    # 3단 (Z: 1.6) - Box (90도 회전 및 마찰력 적용)
    # --------------------------------------------------------------------------
    if not should_skip(r_pos, 1.6, "+X-Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 1.6 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    if not should_skip(r_pos, 1.6, "-X-Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 1.6 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    if not should_skip(r_pos, 1.6, "+X+Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 1.6 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    if not should_skip(r_pos, 1.6, "-X+Y"): 
        box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 1.6 + Z_BASE_OFFSET), r=item_rotation), "box_asset", -1, 0)
        if box_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)

    # --------------------------------------------------------------------------
    # 4단 (Z: 2.2) - Tote (90도 회전 및 마찰력 적용)
    # --------------------------------------------------------------------------
    if not should_skip(r_pos, 2.2, "+X-Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 2.2 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    if not should_skip(r_pos, 2.2, "-X-Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my - MOVE_RACK_Y_OFFSET, 2.2 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    if not should_skip(r_pos, 2.2, "+X+Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx + MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 2.2 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

    if not should_skip(r_pos, 2.2, "-X+Y"): 
        tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(mx - MOVE_RACK_X_OFFSET, my + MOVE_RACK_Y_OFFSET, 2.2 + Z_BASE_OFFSET), r=item_rotation), "tote_asset", -1, 0)
        if tote_handle != -1:
            shape_props = gym.get_actor_rigid_shape_properties(env, tote_handle)
            for i in range(len(shape_props)):
                shape_props[i].friction = ITEM_FRICTION
                shape_props[i].rolling_friction = ITEM_ROLLING_FRICTION
                shape_props[i].restitution = ITEM_RESTITUTION
            gym.set_actor_rigid_shape_properties(env, tote_handle, shape_props)

# ==============================================================================
# 2. LOCK RACK (고정식 선반 - 회전 없음) 및 적재물 생성
# ==============================================================================
lock_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.570796)

for lx, ly, lz in lock_rack_positions:
    r_pos = (lx, ly, lz)
    lock_rack_handle = gym.create_actor(env, lock_rack_asset, gymapi.Transform(p=gymapi.Vec3(lx, ly, lz)), "lock_rack_asset", -1, 0)
    
    # 1단 (Z: 0.4) - Tote
    if not should_skip(r_pos, 0.4, "-X"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(lx - LOCK_RACK_X_OFFSET, ly, 0.4 + Z_BASE_OFFSET), r=lock_rotation), "tote_asset", -1, 0)
    if not should_skip(r_pos, 0.4, "+X"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(lx + LOCK_RACK_X_OFFSET, ly, 0.4 + Z_BASE_OFFSET), r=lock_rotation), "tote_asset", -1, 0)

    # 2단 (Z: 1.0) - Box
    if not should_skip(r_pos, 1.0, "-X"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(lx - LOCK_RACK_X_OFFSET, ly, 1.0 + Z_BASE_OFFSET), r=lock_rotation), "box_asset", -1, 0)
    if not should_skip(r_pos, 1.0, "+X"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(lx + LOCK_RACK_X_OFFSET, ly, 1.0 + Z_BASE_OFFSET), r=lock_rotation), "box_asset", -1, 0)

    # 3단 (Z: 1.6) - Tote
    if not should_skip(r_pos, 1.6, "-X"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(lx - LOCK_RACK_X_OFFSET, ly, 1.6 + Z_BASE_OFFSET), r=lock_rotation), "tote_asset", -1, 0)
    if not should_skip(r_pos, 1.6, "+X"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(lx + LOCK_RACK_X_OFFSET, ly, 1.6 + Z_BASE_OFFSET), r=lock_rotation), "tote_asset", -1, 0)

    # 4단 (Z: 2.2) - Box
    if not should_skip(r_pos, 2.2, "-X"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(lx - LOCK_RACK_X_OFFSET, ly, 2.2 + Z_BASE_OFFSET), r=lock_rotation), "box_asset", -1, 0)
    if not should_skip(r_pos, 2.2, "+X"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(lx + LOCK_RACK_X_OFFSET, ly, 2.2 + Z_BASE_OFFSET), r=lock_rotation), "box_asset", -1, 0)


# ==============================================================================
# 3. LOCK RACK (자체 회전된 선반 2개) 및 적재물 생성 (박스/토트 회전 없음)
# ==============================================================================
rack_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.570796)

for rx, ry, rz in rotated_lock_rack_positions:
    r_pos = (rx, ry, rz)
    lock_rack_handle = gym.create_actor(env, lock_rack_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry, rz), r=rack_rotation), "lock_rack_asset", -1, 0)
    
    # 1단 (Z: 0.4) - Tote
    if not should_skip(r_pos, 0.4, "-Y"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry - LOCK_RACK_X_OFFSET, 0.4 + Z_BASE_OFFSET)), "tote_asset", -1, 0)
    if not should_skip(r_pos, 0.4, "+Y"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry + LOCK_RACK_X_OFFSET, 0.4 + Z_BASE_OFFSET)), "tote_asset", -1, 0)

    # 2단 (Z: 1.0) - Box
    if not should_skip(r_pos, 1.0, "-Y"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry - LOCK_RACK_X_OFFSET, 1.0 + Z_BASE_OFFSET)), "box_asset", -1, 0)
    if not should_skip(r_pos, 1.0, "+Y"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry + LOCK_RACK_X_OFFSET, 1.0 + Z_BASE_OFFSET)), "box_asset", -1, 0)

    # 3단 (Z: 1.6) - Tote
    if not should_skip(r_pos, 1.6, "-Y"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry - LOCK_RACK_X_OFFSET, 1.6 + Z_BASE_OFFSET)), "tote_asset", -1, 0)
    if not should_skip(r_pos, 1.6, "+Y"): tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry + LOCK_RACK_X_OFFSET, 1.6 + Z_BASE_OFFSET)), "tote_asset", -1, 0)

    # 4단 (Z: 2.2) - Box
    if not should_skip(r_pos, 2.2, "-Y"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry - LOCK_RACK_X_OFFSET, 2.2 + Z_BASE_OFFSET)), "box_asset", -1, 0)
    if not should_skip(r_pos, 2.2, "+Y"): box_handle = gym.create_actor(env, box_asset, gymapi.Transform(p=gymapi.Vec3(rx, ry + LOCK_RACK_X_OFFSET, 2.2 + Z_BASE_OFFSET)), "box_asset", -1, 0)

# =============================================================================================
# --- 로봇 인스턴스 스폰 --- 
# =============================================================================================

low_opts = gymapi.AssetOptions()
low_opts.fix_base_link = False
low_asset = gym.load_asset(sim, asset_root, "urdf/low/v1/low_amr_v1.urdf", low_opts)
# ----------
tote_opts = gymapi.AssetOptions()
tote_opts.fix_base_link = False
tote_asset = gym.load_asset(sim, asset_root, "urdf/lifting_amr/v2/lifting_amr_v2_2.urdf", tote_opts)
# ----------
vacuum_gripper_opts = gymapi.AssetOptions()
vacuum_gripper_opts.fix_base_link, vacuum_gripper_opts.flip_visual_attachments = True, True
vacuum_gripper_opts.armature = 0.01
vacuum_gripper_opts.thickness = 0.001
vacuum_gripper_opts.linear_damping = 0.0
vacuum_gripper_opts.angular_damping = 0.0
vacuum_gripper_opts.override_com = True
vacuum_gripper_opts.override_inertia = True
# 만약 텐서 API 제어를 위해 링크의 강체 관성이 필요하다면 아래 옵션을 켭니다.
vacuum_gripper_opts.override_com = True
vacuum_gripper_opts.override_inertia = True
vacuum_gripper_asset = gym.load_asset(sim, asset_root, "urdf/franka_description/robots/vacuum_gripper_120.urdf", vacuum_gripper_opts)

low_handle = gym.create_actor(env, low_asset, gymapi.Transform(p=gymapi.Vec3(0.0, -2.0, 0.3)), "low_asset", -1, 1)
tote_handle = gym.create_actor(env, tote_asset, gymapi.Transform(p=gymapi.Vec3(0.0, 2.4, 0.3)), "tote_asset", -1, 1)
franka_handle = gym.create_actor(env, vacuum_gripper_asset, gymapi.Transform(p=gymapi.Vec3(-2.0, -2.75, 1.5)), "vacuum_gripper_asset", -1, 0)

small_box_asset = gym.create_box(sim, 0.1, 0.1, 0.11, gymapi.AssetOptions())
small_box_handle = gym.create_actor(env, small_box_asset, gymapi.Transform(p=gymapi.Vec3(-2.0+0.77, -2.75, 1.6)), "small_box", -1, 0)


# ===============================
##### low amr 설정 #####
# ===============================

low_amr = LowAMR(gym, sim, env, low_handle)

waypoint_list = [
    (-1.0, 0.0),    # 1번 목적지
    (2.0, 2.0),    # 2번 목적지
    (-1.5, 3.0),   # 3번 목적지
    (0.0, 0.0)     # 복귀
]
current_wp_idx = 0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 0.2          # 초기 목표 높이를 URDF 한계인 최고점(0.2m)으로 변경
LIFT_THRESHOLD = 0.005     # 가동 범위가 좁으므로 도달 공차를 5mm로 정밀화

# =============================================================================================
# --- LowAMR FSM 상태 정의 및 초기화 --- 
# =============================================================================================


# 상태 제어 관련 타겟 변수 동기화
LIFT_TARGET_POS = 0.16  # 물리 상승 임계 상한선 타겟값
LIFT_THRESHOLD = 0.005   # 도달 확인 오차 공차 (5mm)

# ===============================
##### lifting amr 설정 #####
# ===============================

lifting_amr = LiftingAMR(gym, sim, env, tote_handle)

waypoint_list = [
    (-1.0, 0.0),    # 1번 목적지
    (2.0, 2.0),    # 2번 목적지
    (-1.5, 3.0),   # 3번 목적지
    (0.0, 0.0)     # 복귀
]
current_wp_idx = 0

lift_state = "UP"          # 초기 상태: 상승
lift_target = 2.5          # 초기 목표 높이: 최고점 (1.8m)
LIFT_THRESHOLD = 0.03      # 높이 도달 인정 공차 (3cm)

rot_state = "CCW"             # 초기 상태: 반시계 방향(좌회전) 상승
rot_target = 1.5708           # 초기 목표 각도: 최대 상한선 (+90도)
ROT_THRESHOLD = 0.03          # 각도 도달 인정 공차 (약 1.7도)

fork_state = "EXTEND"       # 초기 상태: 확장
fork_target = 0.7          # 초기 목표 길이: 최대 상한선 (0.7m)
FORK_THRESHOLD = 0.02      # 도달 인정 공차 (2cm)

claw_state = "GRIP"         # 초기 상태: 오므리기
claw_target = 0.0           # 테스트 목표 각도 (라디안)
CLAW_THRESHOLD = 0.02       # 도달 인정 공차 (약 1.1도)

LIFT_AMR_STATE_IDLE             = 10  # 초기 대기 상태 (LowAMR 완료 신호 대기)

# [선반 상자 인양 섹션]
LIFT_AMR_STATE_MOVE_TO_SHELF    = 11  # 1. 선반으로 위치 이동
LIFT_AMR_STATE_LIFT_TO_SHELF    = 12  # 2. 선반으로 높이 이동
LIFT_AMR_STATE_ROT_TO_BOX       = 13  # 3. 선반으로 트레이 회전
LIFT_AMR_STATE_ARM_EXTEND       = 14  # 4. 상자로 트레이 팔 확장
LIFT_AMR_STATE_CLAW_CLOSE       = 15  # 5. 트레이 claw 닫기 (상자 그랩)
LIFT_AMR_STATE_ARM_RETRACT      = 16  # 6. 트레이 팔 수축 (상자 회수)

# [목표 선반 하역 섹션]
LIFT_AMR_STATE_MOVE_TO_TARGET   = 17  # 7. 목표 선반 위치 이동
LIFT_AMR_STATE_LIFT_TO_TARGET   = 18  # 8. 목표 선반 높이 이동
LIFT_AMR_STATE_ROT_TO_TARGET    = 19  # 9. 목표 선반으로 트레이 회전
LIFT_AMR_STATE_ARM_EXTEND_TGT   = 20  # 10. 목표 선반에 트레이 팔 확장
LIFT_AMR_STATE_CLAW_OPEN        = 21  # 11. 트레이 claw 열기 (상자 방출)
LIFT_AMR_STATE_ARM_RETRACT_TGT  = 22  # 12. 트레이 팔 수축 (하역 완료 및 복귀)

LIFT_AMR_STATE_TASK_COMPLETE    = 30  # 모든 공정 완수 및 정지
LIFT_AMR_STATE_MANUAL_TUNING     = 100 # 수동 조정 예외 모드

# 초기 시작 상태 세팅 (LowAMR 주행 시작 시 대기)
tote_current_state = LIFT_AMR_STATE_IDLE

current_tote_lift   = 0.6 if tote_current_state >= LIFT_AMR_STATE_MOVE_TO_SHELF else 0.0
current_tote_rot    = 0.0
current_tote_fork   = 0.0
current_tote_claw   = -0.8

# 현재 tote_handle 스폰 시작 좌표 p=gymapi.Vec3(-0.5, 2.5, 1.0) 기준
# -x 축 방향으로 부드럽게 후진 이동시킬 목표 타겟 좌표 지정 (예: X축 기준 -0.5m 에서 -1.5m 지점으로 후진)
tote_target_x = -0.05
tote_target_y = 2.4

tote_dest_target_x = -2.6
tote_dest_target_y = 2.4

# ===============================
##### franka 설정 #####
# ===============================
franka_ctrl = FrankaController(gym, sim, env, franka_handle, pump_link_name="cobot_pump", scale=1.6)

# ===============================
##### simulation part #####
# ===============================
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(3, 3, 5), gymapi.Vec3(1, 1, 0))

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "tote_forward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "tote_backward")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "tote_next_stage")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "low_left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "low_right")

# 실시간 수동 주행 속도 버퍼 변수
manual_linear_cmd = 0.0
manual_angular_cmd = 0.0

frame_count = 0
state_start_frame = None

low_amr.set_state(0)

target_relative_xyz = [0.7706, -0.0008, 0.0065] # 1.2배 스케일링

shelf_target_pos = (2.8, -0.7)
delivery_pos     = (0.05, 1.2)


robot_state = "MOVE_DOWN_1"    # 초기 상태: 첫 번째 하강
step_dur = 50                 # 각 스테이트별 구동 시간

robot_base_pos = np.array([-2.0, -2.75])
box_pos = np.array([-2.0+0.77, -2.75])

# 베이스에서 박스를 바라보는 상대 벡터 (dx, dy)
delta_pos = box_pos - robot_base_pos

# 💡 1번 관절이 회전해야 할 정확한 라디안 각도 계산 (atan2 활용)
target_box_angle = np.arctan2(delta_pos[1], delta_pos[0])

# ==============================================================================
# 0. 초기화 섹션 (클래스 생성부나 상태 초기화 섹션에 배치하세요)
# ==============================================================================
if not hasattr(franka_ctrl, 'is_holding'):
    # 처음 시작할 때 로봇이 물체를 안 들고 있다고 가정 (0도 위치에서 Pick 예정)
    franka_ctrl.is_holding = False 


while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.clear_lines(viewer)

    # 💡 [키보드 이벤트 핸들러 확장]
    for event in gym.query_viewer_action_events(viewer):
        if tote_current_state == LIFT_AMR_STATE_MANUAL_TUNING:
            if event.value > 0:  
                if event.action == "tote_forward":     manual_linear_cmd = 0.4
                elif event.action == "tote_backward":  manual_linear_cmd = -0.4
                elif event.action == "tote_next_stage":
                    print("\n[수동 제어 완료] SPACE 감지 ➡️ 자동 인양 단계 진입")
                    # lifting_amr.set_twist_velocity(0.0, 0.0) 
                    # tote_current_state = LIFT_AMR_STATE_ARM_EXTEND
            elif event.value == 0:  
                if event.action in ["tote_forward", "tote_backward"]:
                    manual_linear_cmd = 0.0

        # LowAMR 수동 조향 키 맵 수신 (FSM 내부적으로 TASK_COMPLETE 옵션 상태일 때 반영됨)
        if event.value > 0:  
            if event.action == "low_left":    manual_angular_cmd = 1.0   
            elif event.action == "low_right": manual_angular_cmd = -1.0  
        elif event.value == 0:  
            if event.action in ["low_left", "low_right"]:
                manual_angular_cmd = 0.0

    # ===============================

    # -----------------------------------------------------------------
    # 🤖 LowAMR FSM 시퀀스 제어 루프
    # -----------------------------------------------------------------
    # 실시간 모바일 섀시 관절 포지션 동기화 데이터 직접 취득
    
    tote_trigger = low_amr.lift_and_locate_fsm(
        frame_count=frame_count,
        target_shelf=shelf_target_pos,
        delivery_pos=delivery_pos,
        manual_angular_cmd=manual_angular_cmd,
        use_manual_steering=False # 💡 수동 조향 옵션 제어 토글 플래그
    )

    if tote_trigger and tote_current_state == LIFT_AMR_STATE_IDLE: 
        tote_current_state = LIFT_AMR_STATE_MOVE_TO_SHELF
        print("\n[LiftingAMR 발크 신호 수신] ➡️ 1단계 선반 이동 시퀀스 시동!")

    # -----------------------------------------------------------------
    # 🤖 2. LiftingAMR 12단계 FSM 시퀀스 제어 루프 (뼈대 프레임)
    # -----------------------------------------------------------------
    # [초기 상태] LowAMR이 작업을 끝내고 신호를 줄 때까지 제자리 대기
    if tote_current_state == LIFT_AMR_STATE_IDLE:
        lifting_amr.set_twist_velocity(0.0, 0.0)

    # 1. 선반으로 위치 이동
    elif tote_current_state == LIFT_AMR_STATE_MOVE_TO_SHELF:
        tote_arrived = lifting_amr.move_to_point(tote_target_x, tote_target_y, backward=True)

        if tote_arrived: 
            print("[LiftingAMR FSM] 1단계 완료: 선반 앞 정위치 이동 완료 ➡️ 2단계(높이 제어) 진입")
            tote_current_state = LIFT_AMR_STATE_LIFT_TO_SHELF

    # 2. 선반으로 높이 이동
    elif tote_current_state == LIFT_AMR_STATE_LIFT_TO_SHELF:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_height = lifting_amr.get_lift_height()
        current_tote_lift = 0.45
        
        # 💡 각 조인트 도달 판정(abs(현재-목표) < 공차) 조건문이 False 자리에 들어갈 예정입니다.
        if abs(current_height - 0.45) < 0.01:
            print("[LiftingAMR FSM] 2단계 완료: 선반 목표 고도 도달 완료 ➡️ 3단계(트레이 정렬 회전) 진입")
            tote_current_state = LIFT_AMR_STATE_ROT_TO_BOX

    # 3. 선반으로 트레이 회전
    elif tote_current_state == LIFT_AMR_STATE_ROT_TO_BOX:
        current_rot = lifting_amr.get_shuttle_rotation()
        rot_target = 1.5708
        current_tote_rot = rot_target
        claw_target = 0.8 
        current_tote_claw = claw_target
        
        if abs(rot_target - current_rot) < ROT_THRESHOLD:
            print("[LiftingAMR FSM] 3단계 완료: 트레이 셔틀 정렬 회전 완료 ➡️ 4단계(팔 확장 전진) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_EXTEND

    # 4. 상자로 트레이 팔 확장
    elif tote_current_state == LIFT_AMR_STATE_ARM_EXTEND:
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.8
        current_tote_fork = fork_target
        
        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("[LiftingAMR FSM] 4단계 완료: 포크 슬라이더 확장 완료 ➡️ 5단계(Claw 그랩 닫기) 진입")
            tote_current_state = LIFT_AMR_STATE_CLAW_CLOSE

    # 5. 트레이 claw 닫기
    elif tote_current_state == LIFT_AMR_STATE_CLAW_CLOSE:
        current_left_claw, _ = lifting_amr.get_claw_positions()
        claw_target = -0.8 
        current_tote_claw = claw_target

        if abs(claw_target - current_left_claw) < CLAW_THRESHOLD:
            print("[LiftingAMR FSM] 5단계 완료: Claw 상자 그랩 완료 ➡️ 6단계(팔 수축 회수) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_RETRACT

    # 6. 트레이 팔 수축
    elif tote_current_state == LIFT_AMR_STATE_ARM_RETRACT:
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.0 
        current_tote_fork = fork_target

        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("[LiftingAMR FSM] 6단계 완료: 상자 차체 내 회수 완료 ➡️ 7단계(목표 선반 주행) 진입")
            tote_current_state = LIFT_AMR_STATE_MOVE_TO_TARGET

    # 7. 목표 선반 위치 이동
    elif tote_current_state == LIFT_AMR_STATE_MOVE_TO_TARGET:
        # 💡 move_to_point를 쓰지 않고, X축 정렬 및 직선 주행만 독립적으로 수행합니다.
        # 1. X축 이동을 위해 필요한 목표 각도 판별 (최초 1회 고정)
        if lifting_amr.target_yaw_fixed is None:
            curr_p, _ = lifting_amr.get_pose()
            err_x = tote_dest_target_x - curr_p['x']
            print("curr_p['x']: ", curr_p['x'], " tote_dest_target_x: ", tote_dest_target_x, " err_x: ", err_x)
            lifting_amr.target_yaw_fixed = 0.0 if err_x < 0 else np.pi
            lifting_amr.turn_complete = False

        # 2. 목표 각도로 선회 정렬
        if not lifting_amr.turn_complete:
            if lifting_amr.turn_to_yaw(lifting_amr.target_yaw_fixed):
                lifting_amr.turn_complete = True
        
        # 3. 각도 정렬 완료 시 X축 직선 주행 전개
        else:
            # drive_along_axis는 목적지 공차(5cm) 안으로 들어오면 True를 반환합니다.
            x_arrived = lifting_amr.drive_along_axis(
                axis='X', 
                target_coordinate=tote_dest_target_x, 
                direction="BACKWARD"
            )

            if x_arrived: 
                # 💡 다음 단계를 위해 사용했던 주행용 고정 플래그들을 원상복구합니다.
                lifting_amr.target_yaw_fixed = None
                lifting_amr.turn_complete = False
                lifting_amr.grid_move_stage = 0 # 혹시 모를 오동작 방지 리셋

                print("[LiftingAMR FSM] 7단계 완료: X축 하역 목적지 주행 완료 ➡️ 8단계(하역 고도 이동) 진입")
                tote_current_state = LIFT_AMR_STATE_LIFT_TO_TARGET

    # 8. 목표 선반 높이 이동
    elif tote_current_state == LIFT_AMR_STATE_LIFT_TO_TARGET:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_height = lifting_amr.get_lift_height()
        current_tote_lift = 0.45

        if abs(current_height - 0.43) < 0.01:
            print("[LiftingAMR FSM] 8단계 완료: 하역 목표 고도 도달 완료 ➡️ 9단계(하역 방향 회전) 진입")
            tote_current_state = LIFT_AMR_STATE_ROT_TO_TARGET

    # 9. 목표 선반으로 트레이 회전
    elif tote_current_state == LIFT_AMR_STATE_ROT_TO_TARGET:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_rot = lifting_amr.get_shuttle_rotation()
        rot_target = 1.5708
        current_tote_rot = rot_target
        
        if abs(rot_target - current_rot) < ROT_THRESHOLD:
            print("[LiftingAMR FSM] 9단계 완료: 하역 슬롯 정렬 회전 완료 ➡️ 10단계(하역 팔 확장) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_EXTEND_TGT

    # 10. 목표 선반에 트레이 팔 확장
    elif tote_current_state == LIFT_AMR_STATE_ARM_EXTEND_TGT:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.8
        current_tote_fork = fork_target
        claw_target = 0.8 
        current_tote_claw = claw_target

        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("[LiftingAMR FSM] 10단계 완료: 하역지 팔 전진 완료 ➡️ 11단계(Claw 개방 안착) 진입")
            tote_current_state = LIFT_AMR_STATE_CLAW_OPEN

    # 11. 트레이 claw 열기
    elif tote_current_state == LIFT_AMR_STATE_CLAW_OPEN:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_left_claw, _ = lifting_amr.get_claw_positions()
        claw_target = 0.8 
        current_tote_claw = claw_target

        if abs(claw_target - current_left_claw) < CLAW_THRESHOLD:
            print("[LiftingAMR FSM] 11단계 완료: Claw 개방 및 상자 안착 완료 ➡️ 12단계(빈 팔 회수 수축) 진입")
            tote_current_state = LIFT_AMR_STATE_ARM_RETRACT_TGT

    # 12. 트레이 팔 수축
    elif tote_current_state == LIFT_AMR_STATE_ARM_RETRACT_TGT:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_fork = lifting_amr.get_fork_extension()
        fork_target = 0.0 
        current_tote_fork = fork_target

        if abs(fork_target - current_fork) < FORK_THRESHOLD:
            print("\n================================================================")
            print("[LiftingAMR 공정 완료] 12단계 모든 인양/하역 시퀀스가 안전하게 완수되었습니다.")
            print("================================================================")
            tote_current_state = LIFT_AMR_STATE_TASK_COMPLETE

    # [최종 종료] 완수 후 완전 제동 정지 대기
    elif tote_current_state == LIFT_AMR_STATE_TASK_COMPLETE:
        lifting_amr.set_twist_velocity(0.0, 0.0)
        current_height = lifting_amr.get_lift_height()
        current_tote_lift = 0

    # [수동 조정] 키보드 이벤트를 통한 미세 디버깅 튜닝 세션
    elif tote_current_state == LIFT_AMR_STATE_MANUAL_TUNING:
        if manual_linear_cmd > 0:
            lifting_amr.set_twist_velocity(linear_x=manual_linear_cmd, angular_z=0.0)
        elif manual_linear_cmd < 0:
            lifting_amr.set_twist_velocity(linear_x=manual_linear_cmd, angular_z=0.0)
        else:
            lifting_amr.set_twist_velocity(0.0, 0.0)

    # 🎯 LiftingAMR 전용 독립 액추에이터 휠/관절 명령 버퍼 일괄 인젝션
    lifting_amr.set_lift_height(current_tote_lift)
    lifting_amr.set_shuttle_rotation(current_tote_rot)
    lifting_amr.set_fork_extension(current_tote_fork)
    lifting_amr.set_claw_grip(current_tote_claw)
    lifting_amr.apply_actuator_commands()

    ######

    # 💡 [중요] 흡착된 물체가 있다면 물리 시뮬레이션 직후 구체 중심으로 포즈를 갱신해 줍니다.
    franka_ctrl.update_snapping_object()
	
    # ------------------------------------------------------------------
    # 🔄 로봇 상태 머신 (State Machine) - 인자 없이 깨끗하게 동작
    # ------------------------------------------------------------------
    
    
    # ==============================================================================
    # FSM 메인 시퀀스 루프
    # ==============================================================================

    # 1. [0도 위치] 하강 시퀀스 (Pick 또는 Place 토글)
    if robot_state == "MOVE_DOWN_1":
        if franka_ctrl.move_down_sequence(step_duration=step_dur):
            rel_xyz = franka_ctrl.get_sphere_position_relative_to_base()
            if rel_xyz is not None:
                print(f"🎯 0도 하강 완료! 베이스 기준 상대 좌표 -> X: {rel_xyz[0]:.4f}, Y: {rel_xyz[1]:.4f}, Z: {rel_xyz[2]:.4f}")
            
            # 💡 [Toggle 로직] 물체를 안 들고 있다면 0도에서 집어 올립니다 (Pick)
            if not franka_ctrl.is_holding:
                franka_ctrl.attach(small_box_handle, distance_threshold=0.15)
                franka_ctrl.is_holding = True
                print("[Franka Action] 0도 위치에서 물체 부착(Attach) 완료 ➡️ 상승")
            # 💡 [Toggle 로직] 90도에서 물체를 들고 왔다면 여기선 내려놓습니다 (Place)
            else:
                franka_ctrl.detach()
                franka_ctrl.is_holding = False
                print("[Franka Action] 0도 위치에 물체 탈착(Detach) 완료 ➡️ 상승")

            robot_state = "MOVE_UP_1"
            print("[상태 전환] 0도 위치 하강 작업 완수 ➡️ STATE_MOVE_UP_1 단계 진입")
            
    # 2. [0도 위치] 상승 시퀀스
    elif robot_state == "MOVE_UP_1":
        if franka_ctrl.move_up_sequence(step_duration=step_dur):
            robot_state = "ROTATE_90"
            print("[상태 전환] 0도 위치 상승 완료 ➡️ 90도 회전 시작")

    # 3. [회전] 1번 관절을 90도(약 1.571 라디안)로 회전 및 대기
    elif robot_state == "ROTATE_90":
        rot_frame = franka_ctrl.rotate_link1(1.571)
        if rot_frame >= (step_dur + franka_ctrl.pause_duration):
            robot_state = "MOVE_DOWN_2"
            print("[상태 전환] 90도 회전 완료 ➡️ 90도 위치 하강 시작")

    # 4. [90도 위치] 하강 시퀀스 (Place 또는 Pick 토글)
    elif robot_state == "MOVE_DOWN_2":
        if franka_ctrl.move_down_sequence(step_duration=step_dur):
            
            # 💡 [Toggle 로직] 0도에서 물체를 들고 왔다면 90도에 내려놓습니다 (Place)
            if franka_ctrl.is_holding:
                franka_ctrl.detach()
                franka_ctrl.is_holding = True
                print("[Franka Action] 90도 위치에 물체 탈착(Detach) 완료 ➡️ 상승")
            # 💡 [Toggle 로직] 0도로 빈손 복귀 후 90도에 물체가 있다면 다시 집어 올립니다 (Pick)
            else:
                franka_ctrl.attach(small_box_handle, distance_threshold=0.15)
                franka_ctrl.is_holding = False
                print("[Franka Action] 90도 위치에서 물체 부착(Attach) 완료 ➡️ 상승")

            robot_state = "MOVE_UP_2"
            print("[상태 전환] 90도 위치 하강 작업 완수 ➡️ STATE_MOVE_UP_2 단계 진입")

    # 5. [90도 위치] 상승 시퀀스
    elif robot_state == "MOVE_UP_2":
        if franka_ctrl.move_up_sequence(step_duration=step_dur):
            robot_state = "ROTATE_RETURN"
            print("[상태 전환] 90도 위치 상승 완료 ➡️ 0도 원위치 복귀 회전 시작")

    # 6. [원위치 복귀 회전] 1번 관절을 다시 0도로 돌려놓기
    elif robot_state == "ROTATE_RETURN":
        rot_frame = franka_ctrl.rotate_link1(0.0)
        if rot_frame >= (step_dur + franka_ctrl.pause_duration):
            robot_state = "MOVE_DOWN_1"
            # 현재 들고 있는지 여부에 따라 다음 사이클의 로그 분기 출력
            job_type = "배달하러 이동" if franka_ctrl.is_holding else "가져오러 빈손 이동"
            print(f"[🔄 무한 순환] 원위치 복귀 완료 ({job_type}) ➡️ 시퀀스 처음부터 재시작\n")

    franka_ctrl.draw_ee_debug_sphere(viewer, radius=0.02, color=(1.0, 0.0, 0.0))

    franka_ctrl.draw_z_rotational_circle(
        viewer, 
        target_relative_xyz, 
        num_spheres=20, 
        sphere_radius=0.04, 
        color=(0.0, 0.5, 1.0)  # 스카이 블루 색상
    )

    gym.sync_frame_time(sim)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)