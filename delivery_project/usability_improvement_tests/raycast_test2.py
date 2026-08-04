"""
Isaac Gym: 3D 뷰포트 마우스 클릭 -> 발사체(projectile) 발사 방식 피킹 예제 (v9)
================================================================================

v8 -> v9 변경 사항
-------------------
- 박스 명중 시 색상 처리를 "한 번 맞으면 계속 하이라이트 유지"에서
  "맞을 때마다 기본색 <-> 하이라이트색을 토글"하는 방식으로 변경했습니다.
  (box_hit_state 딕셔너리로 각 박스의 현재 하이라이트 여부를 추적)
- 디바운스: 같은 발사체가 같은 프레임에 여러 번 접촉 이벤트를 만들어도
  hit_slot["active"] 체크로 인해 그 발사체는 단 한 번만 토글에 반영됩니다
  (발사체가 명중과 동시에 비활성화되어 풀로 반납되므로).

실행
----
python isaacgym_mouse_pick_example.py
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil


MAX_PROJECTILES = 30
LAUNCH_SPEED = 8.0
LAUNCH_OFFSET = 0.3
PROJECTILE_LIFETIME_FRAMES = 180  # 약 3초(60fps 기준) 후 자동으로 풀에 반납
PARK_POSITION = (0.0, 0.0, -50.0)

VIEWER_HORIZONTAL_FOV_DEG = 90.0

PROJECTILE_COLLISION_FILTER = 1  # 박스(filter=0)와는 충돌하되 발사체끼리는 충돌하지 않도록 분리


def main():
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Isaac Gym mouse-click projectile picking example")

    # ---------------------------------------------------------------
    # 1. 시뮬레이션 생성
    # ---------------------------------------------------------------
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
    )
    if sim is None:
        raise RuntimeError("시뮬레이션 생성 실패")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ---------------------------------------------------------------
    # 2. 뷰어 생성
    # ---------------------------------------------------------------
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("뷰어 생성 실패 (headless 모드에서는 사용 불가)")

    cam_pos = gymapi.Vec3(2.0, -2.5, 1.8)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # ---------------------------------------------------------------
    # 3. 환경 및 타겟(박스 3개) 생성
    # ---------------------------------------------------------------
    spacing = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    box_options = gymapi.AssetOptions()
    box_options.fix_base_link = True
    box_asset = gym.create_box(sim, 0.2, 0.2, 0.2, box_options)

    default_color = gymapi.Vec3(0.6, 0.6, 0.6)
    highlight_color = gymapi.Vec3(1.0, 0.2, 0.2)

    target_handles = []
    positions = [(-0.6, 0.0, 0.5), (0.0, 0.0, 0.5), (0.6, 0.0, 0.5)]
    for i, (x, y, z) in enumerate(positions):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(x, y, z)
        handle = gym.create_actor(env, box_asset, pose, f"box_{i}", 0, 0)  # group=0, filter=0
        gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL, default_color)
        target_handles.append(handle)

    # 각 박스의 현재 하이라이트 상태 (False=기본색, True=하이라이트색)
    box_hit_state = {h: False for h in target_handles}

    # ---------------------------------------------------------------
    # 4. 발사체 애셋 + 오브젝트 풀
    #    (같은 좌표에 겹쳐 파킹해도 서로 밀어내지 않도록 collision filter 분리)
    # ---------------------------------------------------------------
    proj_options = gymapi.AssetOptions()
    proj_options.density = 1000.0
    proj_options.disable_gravity = True
    projectile_asset = gym.create_sphere(sim, 0.08, proj_options)
    projectile_color = gymapi.Vec3(1.0, 0.0, 1.0)

    projectile_pool = []  # 각 원소: {"handle": h, "active": bool, "age": int}
    for i in range(MAX_PROJECTILES):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*PARK_POSITION)
        handle = gym.create_actor(
            env, projectile_asset, pose, f"proj_{i}",
            0, PROJECTILE_COLLISION_FILTER  # group=0, filter=1 -> 발사체끼리 충돌 안 함
        )
        gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL, projectile_color)
        projectile_pool.append({"handle": handle, "active": False, "age": 0})

    # ---------------------------------------------------------------
    # 5. 접촉 판정을 위한 env-local body 인덱스 매핑
    # ---------------------------------------------------------------
    def env_body_index(actor_handle):
        return gym.get_actor_rigid_body_index(env, actor_handle, 0, gymapi.DOMAIN_ENV)

    target_body_to_handle = {env_body_index(h): h for h in target_handles}
    projectile_body_to_slot = {env_body_index(p["handle"]): p for p in projectile_pool}

    # ---------------------------------------------------------------
    # 6. 마우스 클릭 이벤트 구독
    # ---------------------------------------------------------------
    gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "shoot")

    def screen_to_ray(mouse_x, mouse_y):
        cam_transform = gym.get_viewer_camera_transform(viewer, env)
        origin = np.array([cam_transform.p.x, cam_transform.p.y, cam_transform.p.z])

        forward_local = gymapi.Vec3(0.0, 0.0, 1.0)
        right_local = gymapi.Vec3(-1.0, 0.0, 0.0)
        up_local = gymapi.Vec3(0.0, 1.0, 0.0)

        forward = cam_transform.r.rotate(forward_local)
        right = cam_transform.r.rotate(right_local)
        up = cam_transform.r.rotate(up_local)

        forward = np.array([forward.x, forward.y, forward.z])
        right = np.array([right.x, right.y, right.z])
        up = np.array([up.x, up.y, up.z])

        ndc_x = 2.0 * mouse_x - 1.0
        ndc_y = 1.0 - 2.0 * mouse_y

        viewer_size = gym.get_viewer_size(viewer)
        aspect = float(viewer_size.x) / float(viewer_size.y)

        h_fov = math.radians(VIEWER_HORIZONTAL_FOV_DEG)
        half_w = math.tan(h_fov / 2.0)
        half_h = half_w / aspect

        direction = forward + right * ndc_x * half_w + up * ndc_y * half_h
        direction = direction / np.linalg.norm(direction)

        return origin, direction

    def fire_projectile(origin, direction):
        slot = next((p for p in projectile_pool if not p["active"]), None)
        if slot is None:
            slot = projectile_pool[0]  # 전부 사용 중이면 0번을 강제 재사용

        spawn_pos = origin + direction * LAUNCH_OFFSET

        body_states = gym.get_actor_rigid_body_states(env, slot["handle"], gymapi.STATE_ALL)
        body_states["pose"]["p"].fill((float(spawn_pos[0]), float(spawn_pos[1]), float(spawn_pos[2])))
        body_states["vel"]["linear"][0] = (
            float(direction[0]) * LAUNCH_SPEED,
            float(direction[1]) * LAUNCH_SPEED,
            float(direction[2]) * LAUNCH_SPEED,
        )
        body_states["vel"]["angular"].fill((0.0, 0.0, 0.0))
        gym.set_actor_rigid_body_states(env, slot["handle"], body_states, gymapi.STATE_ALL)

        slot["active"] = True
        slot["age"] = 0
        return slot

    def toggle_box_color(target_handle):
        """맞을 때마다 기본색 <-> 하이라이트색을 토글."""
        is_highlighted = box_hit_state[target_handle]
        new_state = not is_highlighted
        new_color = highlight_color if new_state else default_color
        gym.set_rigid_body_color(env, target_handle, 0, gymapi.MESH_VISUAL, new_color)
        box_hit_state[target_handle] = new_state

    active_slots = []

    # ---------------------------------------------------------------
    # 7. 메인 루프
    # ---------------------------------------------------------------
    print("왼쪽 마우스 버튼으로 클릭하면 그 방향으로 발사체가 날아갑니다. (창을 닫으면 종료)", flush=True)

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "shoot" and evt.value > 0:
                mouse_pos = gym.get_viewer_mouse_position(viewer)
                origin, direction = screen_to_ray(mouse_pos.x, mouse_pos.y)
                slot = fire_projectile(origin, direction)
                if slot not in active_slots:
                    active_slots.append(slot)

        # --- get_env_rigid_contacts() 기반 실제 접촉 판정 ---
        contacts = gym.get_env_rigid_contacts(env)

        for c in contacts:
            b0 = int(c["body0"])
            b1 = int(c["body1"])

            if b0 == -1 or b1 == -1:
                continue  # 바닥 평면 또는 미인식 접촉은 제외

            hit_target = None
            hit_slot = None
            if b0 in projectile_body_to_slot and b1 in target_body_to_handle:
                hit_slot = projectile_body_to_slot[b0]
                hit_target = target_body_to_handle[b1]
            elif b1 in projectile_body_to_slot and b0 in target_body_to_handle:
                hit_slot = projectile_body_to_slot[b1]
                hit_target = target_body_to_handle[b0]

            if hit_target is not None and hit_slot is not None and hit_slot["active"]:
                toggle_box_color(hit_target)
                print(f"명중: {gym.get_actor_name(env, hit_target)} "
                      f"-> {'하이라이트' if box_hit_state[hit_target] else '기본색'}")
                hit_slot["active"] = False  # 맞은 발사체는 풀로 반납 (중복 토글 방지)

        # --- 수명이 다한 발사체 정리 ---
        still_active = []
        for slot in active_slots:
            if not slot["active"]:
                continue
            slot["age"] += 1
            if slot["age"] > PROJECTILE_LIFETIME_FRAMES:
                slot["active"] = False
            else:
                still_active.append(slot)
        active_slots = still_active

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()