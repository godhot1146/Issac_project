"""
raycast_picker.py
==================
Isaac Gym 뷰어에서 마우스 클릭으로 오브젝트를 피킹(선택)하는 모듈.

색상을 바꾸는 대신, 선택된 오브젝트를 감싸는 큰 와이어프레임 구체를
토글로 표시/숨김합니다. 원본 색상/텍스처는 전혀 건드리지 않습니다.

사용법
------
    picker = RaycastPicker(gym, sim, viewer, env)

    picker.register_sphere(arm_handle, "indy7_arm_1", radius=0.6, marker_radius=0.75)
    picker.register_obb(wall_handle, "wall_back", half_extent=(0.1, 4.25, 2.0))

    gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "pick_object")

    # 메인 루프 이벤트 처리 부분에서:
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "pick_object" and evt.value > 0:
            picker.handle_click()

    # 메인 루프에서 매 프레임, gym.clear_lines() 이후 / gym.draw_viewer() 이전에 호출:
    picker.draw_selection_markers()
"""

import math
import numpy as np
from isaacgym import gymapi, gymutil


class PickableObject:
    def __init__(self, handle, name, shape, marker_radius,
                 radius=None, half_extent=None, body_index=0):
        self.handle = handle
        self.name = name
        self.shape = shape  # 'sphere' or 'obb'
        self.marker_radius = marker_radius  # 선택 시 그릴 마커 구체의 반지름
        self.radius = radius
        self.half_extent = half_extent  # (hx, hy, hz)
        self.body_index = body_index
        self.is_selected = False


class RaycastPicker:
    def __init__(self, gym, sim, viewer, env, horizontal_fov_deg=90.0,
                 marker_color=(1.0, 1.0, 0.0)):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.env = env
        self.horizontal_fov_deg = horizontal_fov_deg
        self.marker_color = marker_color
        self.objects = []  # list[PickableObject]
        self.on_pick_callback = None  # optional: fn(PickableObject or None)

    # -----------------------------------------------------------
    # 등록
    # -----------------------------------------------------------
    def register_sphere(self, handle, name, radius, marker_radius=None, body_index=0):
        """radius: 레이 교차 판정에 쓰는 근사 반지름.
        marker_radius: 선택 시 표시할 마커 구체 반지름 (기본: radius * 1.3, 물체를 넉넉히 감싸도록)."""
        marker_radius = marker_radius if marker_radius is not None else radius * 1.3
        obj = PickableObject(handle, name, "sphere", marker_radius,
                              radius=radius, body_index=body_index)
        self.objects.append(obj)
        return obj

    def register_obb(self, handle, name, half_extent, marker_radius=None, body_index=0):
        """half_extent: (hx, hy, hz) 박스 절반 크기.
        marker_radius: 기본값은 박스를 완전히 감싸는 외접구 반지름(대각선 길이의 절반)."""
        half_extent = tuple(half_extent)
        if marker_radius is None:
            marker_radius = math.sqrt(sum(h * h for h in half_extent))
        obj = PickableObject(handle, name, "obb", marker_radius,
                              half_extent=half_extent, body_index=body_index)
        self.objects.append(obj)
        return obj

    def unregister(self, handle):
        self.objects = [o for o in self.objects if o.handle != handle]

    # -----------------------------------------------------------
    # 레이 계산 (카메라 forward=+Z, right=-X, up=+Y 로컬 축 컨벤션)
    # -----------------------------------------------------------
    def screen_to_ray(self, mouse_x, mouse_y):
        cam_transform = self.gym.get_viewer_camera_transform(self.viewer, self.env)
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

        viewer_size = self.gym.get_viewer_size(self.viewer)
        aspect = float(viewer_size.x) / float(viewer_size.y)

        h_fov = math.radians(self.horizontal_fov_deg)
        half_w = math.tan(h_fov / 2.0)
        half_h = half_w / aspect

        direction = forward + right * ndc_x * half_w + up * ndc_y * half_h
        direction = direction / np.linalg.norm(direction)

        return origin, direction

    # -----------------------------------------------------------
    # 교차 판정
    # -----------------------------------------------------------
    def _get_body_pose(self, obj):
        state = self.gym.get_actor_rigid_body_states(
            self.env, obj.handle, gymapi.STATE_POS
        )["pose"][obj.body_index]
        p, r = state["p"], state["r"]
        pos = gymapi.Vec3(float(p["x"]), float(p["y"]), float(p["z"]))
        rot = gymapi.Quat(float(r["x"]), float(r["y"]), float(r["z"]), float(r["w"]))
        return pos, rot

    @staticmethod
    def _quat_conjugate(q):
        return gymapi.Quat(-q.x, -q.y, -q.z, q.w)

    def _ray_intersect_sphere(self, ray_origin, ray_dir, center, radius):
        oc = ray_origin - center
        b = np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - radius * radius
        disc = b * b - c
        if disc < 0:
            return False, None
        sqrt_disc = math.sqrt(disc)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        t_hit = t1 if t1 >= 0 else t2
        if t_hit < 0:
            return False, None
        return True, t_hit

    def _ray_intersect_obb(self, ray_origin, ray_dir, box_pos, box_quat, half_extent):
        inv_rot = self._quat_conjugate(box_quat)

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
        he = half_extent

        t_min, t_max = -np.inf, np.inf
        for axis in range(3):
            if abs(ld[axis]) < 1e-9:
                if lo[axis] < -he[axis] or lo[axis] > he[axis]:
                    return False, None
                continue
            t1 = (-he[axis] - lo[axis]) / ld[axis]
            t2 = (he[axis] - lo[axis]) / ld[axis]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, None

        if t_max < 0:
            return False, None
        t_hit = t_min if t_min >= 0 else t_max
        return True, t_hit

    def _intersect(self, obj, origin, direction):
        pos, rot = self._get_body_pose(obj)
        if obj.shape == "sphere":
            center = np.array([pos.x, pos.y, pos.z])
            return self._ray_intersect_sphere(origin, direction, center, obj.radius)
        else:  # 'obb'
            return self._ray_intersect_obb(origin, direction, pos, rot, obj.half_extent)

    # -----------------------------------------------------------
    # 피킹 + 선택 토글
    # -----------------------------------------------------------
    def pick(self, mouse_x, mouse_y):
        origin, direction = self.screen_to_ray(mouse_x, mouse_y)

        closest_obj = None
        closest_t = np.inf
        for obj in self.objects:
            hit, t = self._intersect(obj, origin, direction)
            if hit and t < closest_t:
                closest_t = t
                closest_obj = obj
        return closest_obj

    def toggle_selection(self, obj):
        obj.is_selected = not obj.is_selected

    def handle_click(self):
        """마우스 클릭 이벤트 발생 시 호출: 피킹 -> 선택 토글 -> 콜백/로그."""
        mouse_pos = self.gym.get_viewer_mouse_position(self.viewer)
        hit_obj = self.pick(mouse_pos.x, mouse_pos.y)

        if hit_obj is not None:
            self.toggle_selection(hit_obj)
            print(f"[피킹] 클릭됨: {hit_obj.name} "
                  f"-> {'선택됨' if hit_obj.is_selected else '선택 해제'}")
        else:
            print("[피킹] 클릭됐지만 등록된 오브젝트에 맞지 않음")

        if self.on_pick_callback is not None:
            self.on_pick_callback(hit_obj)

        return hit_obj

    # -----------------------------------------------------------
    # 마커 그리기 (매 프레임 호출 필요)
    # -----------------------------------------------------------
    def draw_selection_markers(self):
        """선택된 모든 오브젝트를 감싸는 와이어프레임 구체를 그림.
        gym.clear_lines(viewer) 이후, gym.draw_viewer() 이전에 매 프레임 호출해야 함
        (디버그 라인은 프레임마다 초기화되므로)."""
        for obj in self.objects:
            if not obj.is_selected:
                continue
            pos, _ = self._get_body_pose(obj)
            sphere_geom = gymutil.WireframeSphereGeometry(
                obj.marker_radius, 16, 16, gymapi.Transform(), color=self.marker_color
            )
            pose = gymapi.Transform(p=pos)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env, pose)