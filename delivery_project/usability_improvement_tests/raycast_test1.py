import math
from isaacgym import gymapi, gymutil

# ----------------------------
# 1. 기본 초기화
# ----------------------------
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("시뮬레이션 생성 실패")

# 바닥 평면
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 환경 하나 생성 (디버그 그리기용으로 env 핸들 필요)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# ----------------------------
# 2. 뷰어 생성
# ----------------------------
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("뷰어 생성 실패")

# 카메라 초기 위치 설정 (원하는 대로 조정 가능)
cam_pos = gymapi.Vec3(3.0, 3.0, 2.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

# ----------------------------
# 3. 스페이스바 키 이벤트 구독
# ----------------------------
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "draw_spheres")

# ----------------------------
# 4. 디버그 구체를 그릴 상태 저장
# ----------------------------
NUM_SPHERES = 10
SPACING = 0.10  # 10cm
SPHERE_RADIUS = 0.02
sphere_color = (1.0, 0.0, 0.0)

sphere_positions = []  # 스페이스바 눌렀을 때 계산된 위치들을 저장

def compute_sphere_positions():
    """뷰어 카메라의 현재 위치/방향을 기준으로 10cm 간격 구체 위치 계산"""
    cam_transform = gym.get_viewer_camera_transform(viewer, env)
    origin = cam_transform.p

    # 카메라 로컬 -Z 축이 카메라가 바라보는 forward 방향
    # (반대로 보이면 (0,0,1)로 바꿔서 부호를 뒤집으세요)
    forward_local = gymapi.Vec3(0.0, 0.0, 1.0)
    forward = cam_transform.r.rotate(forward_local)

    positions = []
    for i in range(1, NUM_SPHERES + 1):
        dist = SPACING * i
        pos = gymapi.Vec3(
            origin.x + forward.x * dist,
            origin.y + forward.y * dist,
            origin.z + forward.z * dist,
        )
        positions.append(pos)
    return positions

# ----------------------------
# 5. 메인 루프
# ----------------------------
while not gym.query_viewer_has_closed(viewer):

    # 키 이벤트 처리
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "draw_spheres" and evt.value > 0:
            sphere_positions = compute_sphere_positions()
            print(f"스페이스바 입력: 구체 {len(sphere_positions)}개 위치 갱신")

    # 물리 스텝
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 디버그 라인 초기화 (매 프레임 지우고 다시 그림)
    gym.clear_lines(viewer)

    # 저장된 위치에 구체 그리기
    for pos in sphere_positions:
        sphere_geom = gymutil.WireframeSphereGeometry(
            SPHERE_RADIUS, 8, 8, gymapi.Transform(), color=sphere_color
        )
        pose = gymapi.Transform(p=pos)
        gymutil.draw_lines(sphere_geom, gym, viewer, env, pose)

    # 상태 업데이트 및 렌더링
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)