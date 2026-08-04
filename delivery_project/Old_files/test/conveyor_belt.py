"""
ConveyorBelt
============

Isaac Gym에서는 컨베이어 벨트 "메시" 자체를 물리적으로 이동시킬 수 없다.
따라서 이 클래스는:
  1) 벨트 위에 올라간 물체(박스 등)에 매 프레임 속도를 주입해서 실제로 밀려가게 하고,
  2) 벨트 표면의 텍스처 프레임을 스크롤해서 시각적으로 "벨트가 움직이는 것처럼" 보이게 한다.

사용 흐름:
    conveyor = ConveyorBelt(gym, sim, env, conveyor_handle,
                             belt_link_name="belt_link",
                             texture_frame_paths=[...],
                             speed=0.5)

    # gym.prepare_sim(sim) 이후:
    conveyor.setup_tensors(belt_length=3.5)

    # 벨트 위에 올라갈 물체 등록 (여러 개 가능)
    conveyor.register_item(box_manager1.handle)
    conveyor.register_item(box_manager2.handle)

    # 동작 트리거 (예: 로봇 작업 완료 시점)
    conveyor.move_backward(distance=1.2)   # 로컬 -Y 방향으로 1.2m만 이동 후 자동 정지
    # 또는
    conveyor.move_forward()                # distance 생략 시 stop() 호출 전까지 무한 이동

    # 메인 루프에서 매 프레임:
    conveyor.update(frame_count)

    # 로봇이 물체를 붙잡는 동안에는 컨베이어 힘이 가해지지 않도록:
    conveyor.set_item_grabbed(box_manager1.handle, arm.is_attached and arm.attached_handle == box_manager1.handle)
"""

import torch
from isaacgym import gymapi, gymtorch
from scipy.spatial.transform import Rotation as R


def _world_to_local(origin_pos, origin_rot, world_pos):
    """world_pos를 origin(위치/회전) 기준 로컬 좌표로 변환."""
    delta = gymapi.Vec3(world_pos.x - origin_pos.x,
                         world_pos.y - origin_pos.y,
                         world_pos.z - origin_pos.z)
    return origin_rot.inverse().rotate(delta)


def _flatten_keep_yaw(rot: gymapi.Quat) -> gymapi.Quat:
    """roll/pitch만 0으로 눌러 수평 정렬하고, yaw(z축 회전)는 현재 값 그대로 유지."""
    q = R.from_quat([rot.x, rot.y, rot.z, rot.w])
    _, _, yaw = q.as_euler('xyz', degrees=False)
    flat = R.from_euler('z', yaw)
    fx, fy, fz, fw = flat.as_quat()
    return gymapi.Quat(fx, fy, fz, fw)


class ConveyorBelt:
    def __init__(self, gym, sim, env, conveyor_handle,
                 belt_link_name="belt_link",
                 texture_frame_paths=None,
                 speed=0.5,
                 dt=1.0 / 60.0):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = conveyor_handle
        self.belt_link_name = belt_link_name
        self.speed = speed
        self.dt = dt

        # 텍스처(로컬 인덱스)는 setup_tensors 없이도 바로 조회 가능
        self.belt_rb_local_idx = gym.find_actor_rigid_body_index(
            env, conveyor_handle, belt_link_name, gymapi.DOMAIN_ACTOR)
        if self.belt_rb_local_idx == -1:
            print(f"[ConveyorBelt] 경고: '{belt_link_name}' 링크를 찾지 못했습니다. "
                  f"텍스처 스크롤이 비활성화됩니다.")

        # 텍스처 애니메이션 프레임 로드
        self.texture_handles = []
        if texture_frame_paths:
            for path in texture_frame_paths:
                self.texture_handles.append(gym.create_texture_from_file(sim, path))
        self.num_frames = len(self.texture_handles)
        self.texture_pixel_pos = 0.0
        self.texture_pixels_per_frame = 0.0  # setup_tensors에서 belt_length 알고 나서 계산

        # 동작 상태
        self.running = False
        self.direction = +1              # +1: 정방향(로컬 +Y), -1: 역방향(로컬 -Y)
        self.target_distance = None      # None이면 stop() 호출 전까지 무한 이동
        self._moved_distance = 0.0
        self._distance_ref_item = None   # 이동 거리 측정 기준으로 삼을 아이템 handle

        # 벨트 위 물체 관리: handle -> 상태 dict
        self.items = {}

        # prepare_sim 이후에 채워짐
        self.root_states = None
        self.rb_states = None
        self.belt_rb_sim_idx = None
        self.belt_length = None

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------
    def setup_tensors(self, belt_length=3.5):
        """gym.prepare_sim(sim) 호출 '이후'에 반드시 실행."""
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root_tensor).view(-1, 13)

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor).view(-1, 13)

        self.belt_rb_sim_idx = self.gym.find_actor_rigid_body_index(
            self.env, self.handle, self.belt_link_name, gymapi.DOMAIN_SIM)
        if self.belt_rb_sim_idx == -1:
            # 링크 이름을 못 찾으면 액터의 0번 리지드바디로 폴백
            self.belt_rb_sim_idx = self.gym.get_actor_rigid_body_index(
                self.env, self.handle, 0, gymapi.DOMAIN_SIM)
            print(f"[ConveyorBelt] 경고: belt_rb_sim_idx를 0번 링크로 폴백했습니다. "
                  f"실제 벨트 링크 이름을 확인하세요.")

        self.belt_length = belt_length
        if self.num_frames > 0:
            self.texture_pixels_per_frame = -(self.speed * self.dt * self.num_frames) / self.belt_length

    # ------------------------------------------------------------------
    # 물체(아이템) 등록 / 해제
    # ------------------------------------------------------------------
    def register_item(self, actor_handle, root_link_index=0):
        """컨베이어가 관리할 물체를 등록. 여러 개 등록 가능."""
        actor_sim_idx = self.gym.get_actor_index(self.env, actor_handle, gymapi.DOMAIN_SIM)
        rb_sim_idx = self.gym.get_actor_rigid_body_index(
            self.env, actor_handle, root_link_index, gymapi.DOMAIN_SIM)

        if actor_sim_idx == -1 or rb_sim_idx == -1:
            print(f"[ConveyorBelt] register_item 실패: 유효하지 않은 액터/링크 인덱스")
            return False

        self.items[actor_handle] = {
            "actor_sim_index": actor_sim_idx,
            "rb_sim_index": rb_sim_idx,
            "is_landed": False,
            "prev_z": 999.0,
            "start_local_y": None,
            "grabbed": False,
        }
        return True

    def unregister_item(self, actor_handle):
        """더 이상 컨베이어가 관리하지 않도록 제거 (예: 벨트 끝에서 다른 공정으로 넘어갈 때)."""
        self.items.pop(actor_handle, None)

    def set_item_grabbed(self, actor_handle, grabbed: bool):
        """로봇 등이 물체를 붙잡고 있는 동안에는 컨베이어 힘을 가하지 않도록 표시."""
        if actor_handle in self.items:
            self.items[actor_handle]["grabbed"] = grabbed

    def reset_item_landing(self, actor_handle):
        """물체를 다시 벨트 위에 새로 올려놓았을 때(예: 재사용) 안착 판정을 리셋."""
        if actor_handle in self.items:
            self.items[actor_handle]["is_landed"] = False
            self.items[actor_handle]["prev_z"] = 999.0
            self.items[actor_handle]["start_local_y"] = None

    # ------------------------------------------------------------------
    # 동작 제어 API
    # ------------------------------------------------------------------
    def move_forward(self, distance=None):
        """정방향(로컬 +Y)으로 이동 시작. distance(m)를 주면 그만큼만 이동 후 자동 정지."""
        self._start_run(direction=+1, distance=distance)

    def move_backward(self, distance=None):
        """역방향(로컬 -Y)으로 이동 시작. distance(m)를 주면 그만큼만 이동 후 자동 정지."""
        self._start_run(direction=-1, distance=distance)

    def stop(self):
        """즉시 정지. 물체들의 속도도 0으로 리셋됨(다음 update()에서)."""
        self.running = False
        self.target_distance = None
        self._moved_distance = 0.0
        self._distance_ref_item = None

    @property
    def moved_distance(self):
        return self._moved_distance

    def _start_run(self, direction, distance):
        self.running = True
        self.direction = direction
        self.target_distance = distance
        self._moved_distance = 0.0
        # 거리 측정 기준: 현재 등록된 아이템 중 가장 먼저 등록된 것
        self._distance_ref_item = next(iter(self.items), None)
        for handle, st in self.items.items():
            st["start_local_y"] = None  # update()에서 현재 위치로 새로 기록

    # ------------------------------------------------------------------
    # 매 프레임 업데이트 (메인 루프에서 호출)
    # ------------------------------------------------------------------
    def update(self, frame_count=0):
        if self.root_states is None:
            raise RuntimeError("ConveyorBelt.setup_tensors()를 먼저 호출해야 합니다.")

        belt_pos_t = self.rb_states[self.belt_rb_sim_idx, 0:3]
        belt_rot_t = self.rb_states[self.belt_rb_sim_idx, 3:7]
        belt_r = gymapi.Quat(belt_rot_t[0].item(), belt_rot_t[1].item(),
                              belt_rot_t[2].item(), belt_rot_t[3].item())
        belt_p = gymapi.Vec3(belt_pos_t[0].item(), belt_pos_t[1].item(), belt_pos_t[2].item())

        updated_indices = []

        for handle, st in self.items.items():
            rb_idx = st["rb_sim_index"]
            actor_idx = st["actor_sim_index"]

            pos_t = self.rb_states[rb_idx, 0:3]
            cur_z = pos_t[2].item()
            z_diff = abs(cur_z - st["prev_z"])

            # 낙하 이후 z가 안정화되면 "안착"으로 판정
            if not st["is_landed"] and z_diff < 0.0001 and cur_z > 0.05:
                st["is_landed"] = True

            new_state = self.root_states[actor_idx].clone()
            changed = False

            if self.running and st["is_landed"] and not st["grabbed"]:
                local_vel = gymapi.Vec3(0.0, self.direction * self.speed, 0.0)
                world_vel = belt_r.rotate(local_vel)

                rot_t = self.rb_states[rb_idx, 3:7]
                cur_q = gymapi.Quat(rot_t[0].item(), rot_t[1].item(), rot_t[2].item(), rot_t[3].item())
                flat_q = _flatten_keep_yaw(cur_q)

                new_state[3], new_state[4], new_state[5], new_state[6] = \
                    flat_q.x, flat_q.y, flat_q.z, flat_q.w
                new_state[7], new_state[8], new_state[9] = world_vel.x, world_vel.y, world_vel.z
                new_state[10], new_state[11], new_state[12] = 0.0, 0.0, 0.0
                changed = True

            elif st["is_landed"] and (not self.running or st["grabbed"]):
                # 정지 상태이거나 로봇이 붙잡고 있으면 컨베이어發 속도는 0으로
                new_state[7], new_state[8], new_state[9] = 0.0, 0.0, 0.0
                new_state[10], new_state[11], new_state[12] = 0.0, 0.0, 0.0
                changed = True

            st["prev_z"] = cur_z

            if changed:
                self.root_states[actor_idx] = new_state
                updated_indices.append(actor_idx)

            # 거리 기준 이동이면, 기준 아이템의 이동거리를 측정해서 목표 도달 시 자동 정지
            if (self.running and self.target_distance is not None
                    and handle == self._distance_ref_item and st["is_landed"]):
                world_pos = gymapi.Vec3(pos_t[0].item(), pos_t[1].item(), cur_z)
                local_pos = _world_to_local(belt_p, belt_r, world_pos)
                if st["start_local_y"] is None:
                    st["start_local_y"] = local_pos.y
                self._moved_distance = abs(local_pos.y - st["start_local_y"])
                if self._moved_distance >= self.target_distance:
                    self.stop()

        if updated_indices:
            idx_t = torch.tensor(updated_indices, dtype=torch.int32, device=self.root_states.device)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(idx_t),
                len(updated_indices)
            )

        # 텍스처 스크롤 (가동 중일 때만)
        if self.running and self.num_frames > 0:
            self.texture_pixel_pos += self.texture_pixels_per_frame
            tex_idx = int(self.texture_pixel_pos) % self.num_frames
            self.gym.set_rigid_body_texture(
                self.env, self.handle, self.belt_rb_local_idx, gymapi.MESH_VISUAL,
                self.texture_handles[tex_idx]
            )