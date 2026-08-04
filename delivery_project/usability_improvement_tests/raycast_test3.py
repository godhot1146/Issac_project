"""
Isaac Gym: 3D 뷰포트 마우스 클릭 -> 레이캐스트 기반 오브젝트 피킹 예제 (v10)
================================================================================

v9 -> v10 변경 사항
--------------------
- Isaac Gym Preview Release(레거시) 파이썬 API에는 PhysX 씬 레이캐스트 쿼리
  함수가 노출되어 있지 않습니다(Isaac Sim에만 있는 get_physx_scene_query_interface
  와 달리). 그래서 발사체를 실제로 쏴서 물리 충돌로 판정하는 대신,
  클릭 시 레이-박스(OBB) 교차를 직접 수학적으로 계산해서 즉시 피킹합니다.
- 발사체(projectile) 관련 로직은 모두 제거했습니다. 클릭하면 물리 시뮬레이션
  없이 바로 그 프레임에 어떤 박스가 클릭됐는지 계산되고 색이 토글됩니다.
- ray_intersect_obb(): 레이를 박스의 로컬 좌표계로 변환한 뒤 slab method로
  AABB 교차 검사를 수행합니다. 여러 박스에 동시에 맞으면 가장 가까운 것을
  선택합니다.

실행
----
python isaacgym_mouse_pick_example.py
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil


VIEWER_HORIZONTAL_FOV_DEG = 90.0
BOX_HALF_EXTENT = 0.1  # create_box(sim, 0.2, 0.2, 0.2, ...)와 일치해야 함


def main():
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Isaac Gym mouse-click raycast picking example")

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
        handle = gym.create_actor(env, box_asset, pose, f"box_{i}", 0, 0)
        gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL, default_color)
        target_handles.append(handle)

    # 각 박스의 현재 하이라이트 상태 (False=기본색, True=하이라이트색)
    box_hit_state = {h: False for h in target_handles}

    # ---------------------------------------------------------------
    # 4. 마우스 클릭 이벤트 구독
    # ---------------------------------------------------------------
    gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "pick")

    def screen_to_ray(mouse_x, mouse_y):
        """화면 클릭 좌표(0~1) -> 월드 좌표계 레이(origin, direction)."""
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

    def quat_conjugate(q):
        """단위 쿼터니언의 역회전(켤레)."""
        return gymapi.Quat(-q.x, -q.y, -q.z, q.w)

    def ray_intersect_obb(ray_origin, ray_dir, box_pos, box_quat, half_extent):
        """레이-OBB(회전된 박스) 교차 검사.

        레이를 박스의 로컬 좌표계로 변환한 뒤, 원점 중심의 축정렬 박스(AABB)에
        대한 slab method로 교차 검사를 수행합니다.
        교차하면 (True, t_hit)을, 아니면 (False, None)을 반환합니다.
        t_hit은 ray_origin에서 교차 지점까지의 거리(파라미터)입니다.
        """
        inv_rot = quat_conjugate(box_quat)

        rel = gymapi.Vec3(
            float(ray_origin[0] - box_pos.x),
            float(ray_origin[1] - box_pos.y),
            float(ray_origin[2] - box_pos.z),
        )
        local_origin = inv_rot.rotate(rel)

        dir_vec = gymapi.Vec3(float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2]))
        local_dir = inv_rot.rotate(dir_vec)

        lo = np.array([local_origin.x, local_origin.y, local_origin.z])
        ld = np.array([local_dir.x, local_dir.y, local_dir.z])

        t_min = -np.inf
        t_max = np.inf

        for axis in range(3):
            if abs(ld[axis]) < 1e-9:
                # 레이가 이 축과 평행: 박스 범위 밖이면 교차 불가
                if lo[axis] < -half_extent or lo[axis] > half_extent:
                    return False, None
                continue
            t1 = (-half_extent - lo[axis]) / ld[axis]
            t2 = (half_extent - lo[axis]) / ld[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, None

        if t_max < 0:
            return False, None  # 박스가 레이 뒤쪽에 있음

        t_hit = t_min if t_min >= 0 else t_max
        return True, t_hit

    def get_actor_pose(actor_handle):
        state = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)["pose"][0]
        p = state["p"]
        r = state["r"]
        pos = gymapi.Vec3(float(p["x"]), float(p["y"]), float(p["z"]))
        rot = gymapi.Quat(float(r["x"]), float(r["y"]), float(r["z"]), float(r["w"]))
        return pos, rot

    def pick_at(mouse_x, mouse_y):
        """클릭 좌표로부터 레이를 만들고, 가장 가까운 박스를 찾아 반환 (없으면 None)."""
        origin, direction = screen_to_ray(mouse_x, mouse_y)

        closest_handle = None
        closest_t = np.inf

        for handle in target_handles:
            box_pos, box_quat = get_actor_pose(handle)
            hit, t = ray_intersect_obb(origin, direction, box_pos, box_quat, BOX_HALF_EXTENT)
            if hit and t < closest_t:
                closest_t = t
                closest_handle = handle

        return closest_handle

    def toggle_box_color(target_handle):
        """클릭할 때마다 기본색 <-> 하이라이트색을 토글."""
        is_highlighted = box_hit_state[target_handle]
        new_state = not is_highlighted
        new_color = highlight_color if new_state else default_color
        gym.set_rigid_body_color(env, target_handle, 0, gymapi.MESH_VISUAL, new_color)
        box_hit_state[target_handle] = new_state

    # ---------------------------------------------------------------
    # 5. 메인 루프
    # ---------------------------------------------------------------
    print("왼쪽 마우스 버튼으로 박스를 클릭하면 색이 토글됩니다. (창을 닫으면 종료)", flush=True)

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "pick" and evt.value > 0:
                mouse_pos = gym.get_viewer_mouse_position(viewer)
                hit_handle = pick_at(mouse_pos.x, mouse_pos.y)
                if hit_handle is not None:
                    toggle_box_color(hit_handle)
                    print(f"클릭됨: {gym.get_actor_name(env, hit_handle)} "
                          f"-> {'하이라이트' if box_hit_state[hit_handle] else '기본색'}")
                else:
                    print("클릭됐지만 맞은 박스 없음", flush=True)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()