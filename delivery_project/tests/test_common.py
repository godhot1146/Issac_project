"""
tests/test_common.py

5개 컨트롤러 단독 테스트 스크립트(test_conveyor.py, test_forklift.py, test_lowamr.py,
test_indyarm.py, test_cardboardbox.py)가 공통으로 쓰는 보일러플레이트.

main 스크립트의 Section 1(sim 초기화) / Section 2(최소 바닥) / Section 12-2(물리 스텝 루프)를
그대로 재사용 가능한 함수 형태로 뺀 것이다. 컨트롤러별 차이(에셋, 시나리오)는 각 test_*.py에서
이 모듈의 함수들을 조합해서 구성한다.

주의: ASSET_ROOT는 실제 리포지토리에서 tests/ 폴더가 어디에 위치하느냐에 따라 상대 경로 depth를
조정해야 한다. main.py가 "<repo>/envs/<something>/main.py" 형태였고 asset_root가
"../../assets" 였다면, tests/ 폴더 위치에 맞춰 아래 값을 다시 계산할 것.
"""

import os
from isaacgym import gymapi, gymutil


# ============================================================================
# 경로 설정 — 프로젝트 구조에 맞게 조정
# ============================================================================
ASSET_ROOT = os.environ.get("ISAAC_ASSETS", "/home/henry/Desktop/Issac_asset/isaac_assets")


# ============================================================================
# Sim / Env / Viewer 생성 (main Section 1~2 축소판)
# ============================================================================
def create_minimal_sim(description="Controller Unit Test"):
    """물리엔진 초기화 + 바닥 평면 + 단일 env 생성. 벽/기둥/공장 소품은 생략."""
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description=description)

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.physx.solver_type = 1
    sim_params.physx.use_gpu = True
    sim_params.physx.num_position_iterations = 12
    # num_velocity_iterations는 의도적으로 설정하지 않음. main.py 원본 forklift_steps로
    # 재현했을 때 이 값 없이 정상 회전하는 게 확인됐고(문서 12/13), 반대로 이 값을 다시
    # 넣은 상태(test_common 기반)에서는 회전이 또 안 되는 것도 확인됐다(재현 실험 완료).
    # 즉 이 값 = 8 은 회전을 방해하는 실제 원인 중 하나로 보이므로 넣지 않는다.

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                          args.physics_engine, sim_params)
    gym.set_light_parameters(
        sim, 0,
        gymapi.Vec3(1.5, 1.5, 1.5), gymapi.Vec3(0.4, 0.4, 0.4), gymapi.Vec3(1.0, 0.0, -1.0)
    )

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = -0.05
    gym.add_ground(sim, plane_params)

    # main.py의 room_x=10.0, room_y=8.5와 동일하게 맞춤.
    # (이전 room=6.0일 때 ForkliftAMR 기본 스폰 좌표 x=-3.2가 ±3.0 경계 밖으로 나가 있었음 —
    #  Isaac Gym의 env AABB가 물리적 벽은 아니지만, 컨트롤러의 실제 이동 범위를 항상 여유
    #  있게 감싸도록 main.py 치수를 그대로 쓰는 게 안전하다.)
    room_x, room_y = 10.0, 8.5
    env = gym.create_env(
        sim, gymapi.Vec3(-room_x / 2, -room_y / 2, 0), gymapi.Vec3(room_x / 2, room_y / 2, 3.0), 1
    )

    return gym, sim, env, args


def create_viewer(gym, sim, env, cam_pos=(3.0, -3.0, 2.5), cam_target=(0.0, 0.0, 0.5)):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))
    return viewer


def load_asset(gym, sim, relative_path, fix_base_link=True, density=None,
                replace_cylinder_with_capsule=False):
    """
    asset_root 기준 상대 경로로 URDF 로드. main Section 3/4의 gym.load_asset 패턴과 동일.

    density=None(기본값)이면 AssetOptions.density를 건드리지 않는다 — Isaac Gym 기본값
    (1000.0)이 그대로 적용된다. main.py를 보면 컨베이어/박스/팔 등 고정 소품류는
    density=100.0을 명시하지만, LowAMR/ForkliftAMR 같은 AMR류는 density를 아예 안 건드린다.
    이 구분을 무시하고 전부 100.0으로 통일했다가 ForkliftAMR이 원래보다 10배 가벼워져서
    회전 시 바퀴 그립(최대 마찰력 = 질량×g×마찰계수)이 부족해지는 문제가 있었다.
    -> AMR 컨트롤러(LowAMR, ForkliftAMR)를 로드할 때는 density를 넘기지 말 것.
       고정 소품(박스, 랙, 컨베이어 등)은 main.py처럼 density=100.0을 명시할 것.
    """
    opts = gymapi.AssetOptions()
    opts.fix_base_link = fix_base_link
    if density is not None:
        opts.density = density
    opts.replace_cylinder_with_capsule = replace_cylinder_with_capsule
    return gym.load_asset(sim, ASSET_ROOT, relative_path, opts)


def get_actor_world_pos(gym, env, handle):
    """main 스크립트의 동명 함수와 동일 — 루트 링크 월드 좌표 반환."""
    body_states = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
    pos = body_states['pose']['p'][0]
    return pos['x'], pos['y'], pos['z']


def get_yaw_from_quat(quat):
    """
    참고 코드(Dual-AMR Teleop)의 get_yaw()와 동일. body_states['pose']['r'][0] 같은
    구조화 배열(quat['w'], quat['x'], quat['y'], quat['z'])을 받아 yaw(rad)를 반환한다.
    """
    siny_cosp = 2.0 * (quat['w'] * quat['z'] + quat['x'] * quat['y'])
    cosy_cosp = 1.0 - 2.0 * (quat['y'] ** 2 + quat['z'] ** 2)
    import numpy as np
    return np.arctan2(siny_cosp, cosy_cosp)


# ============================================================================
# 메인 루프 (main Section 12-2 순서 그대로)
# ============================================================================
def main_loop(gym, sim, viewer, on_frame):
    """
    on_frame(frame_count, events) 콜백을 매 프레임 호출하는 공용 루프.
    physics step -> tensor refresh -> on_frame(컨트롤러 처리/러너 update) -> sync_frame_time
    순서는 main 스크립트 Section 12와 동일하게 맞췄다.
    """
    frame_count = 0
    while not gym.query_viewer_has_closed(viewer):
        events = gym.query_viewer_action_events(viewer)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        on_frame(frame_count, events)

        gym.sync_frame_time(sim)
        frame_count += 1

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


# ============================================================================
# 키 입력 헬퍼
# ============================================================================
class HeldKeyTracker:
    """
    Isaac Gym 뷰어 이벤트는 프레임마다 '눌림/뗌' 이벤트만 던져주므로, WASD류 연속 입력을
    처리하려면 마지막 상태를 직접 들고 있어야 한다. event.value > 0 이면 눌림, 0이면 뗌으로 가정.
    (프로젝트의 Isaac Gym 빌드가 뗌 이벤트를 안 보내는 경우, 액션별로 타임아웃 방식으로 바꿔야 함)
    """
    def __init__(self):
        self._held = {}

    def update(self, events):
        for e in events:
            self._held[e.action] = e.value > 0

    def is_held(self, action):
        return self._held.get(action, False)


class ModeToggle:
    """
    TAB 키로 'manual'(원시 메서드 직접 호출) <-> 'scenario'(MotionStep+StepRunner) 모드 전환.
    """
    def __init__(self, gym, viewer, key=gymapi.KEY_TAB, start_mode="manual"):
        gym.subscribe_viewer_keyboard_event(viewer, key, "toggle_test_mode")
        self.mode = start_mode

    def handle_events(self, events):
        for e in events:
            if e.action == "toggle_test_mode" and e.value > 0:
                self.mode = "scenario" if self.mode == "manual" else "manual"
                print(f"[ModeToggle] mode -> {self.mode}")