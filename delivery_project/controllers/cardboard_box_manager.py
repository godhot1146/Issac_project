import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch

class BoxStageRegistry:
    """
    상자의 접기 스테이지를 외부에서 등록/관리하는 레지스트리.
    CardboardBoxManager는 이 레지스트리를 주입받아 사용만 한다.
    """
    def __init__(self):
        self._stages = {}  # name -> {"joint_targets": {...}, "lock_joints": [...]}

    def register(self, name, joint_targets_deg, lock_joints=None, base=None):
        """
        name: 스테이지 식별자 (예: "close_bottom")
        joint_targets_deg: {joint_name: angle_degrees} - 이 스테이지에서 적용할 목표각(도 단위)
        lock_joints: 초기 스폰 시 잠글 관절 이름 리스트 (없으면 잠그지 않음)
        base: 이 스테이지가 이어받을 기존 스테이지 이름.
              지정하면 base의 joint_targets를 먼저 복사한 뒤 새 값으로 덮어씀
              (예: stage_2_close_bottom이 stage_1_fold_sides를 포함하던 것과 동일한 효과)
        """
        targets = {}
        if base is not None:
            if base not in self._stages:
                raise KeyError(f"등록되지 않은 base 스테이지: '{base}'")
            targets.update(self._stages[base]["joint_targets"])
        targets.update(joint_targets_deg)

        self._stages[name] = {
            "joint_targets": targets,
            "lock_joints": list(lock_joints) if lock_joints else [],
        }

    def get(self, name):
        if name not in self._stages:
            raise KeyError(f"등록되지 않은 스테이지: '{name}'")
        return self._stages[name]

    def names(self):
        return list(self._stages.keys())

class CardboardBoxManager:
    """
    골판지 상자(cardboard box) 액터를 생성/제어하고,
    상자 안의 패키지(children) 및 팔레트(platform)와의 부착 관계를 관리하는 클래스.
    """

    # 모든 상자(및 그 안의 자식 패키지)는 하나의 콜리전 필터 그룹을 공유한다.
    # 즉 상자-상자, 상자-자식패키지, 자식패키지-자식패키지 간에는 서로 물리적으로
    # 충돌하지 않고 통과한다
    _SHARED_BOX_FILTER_BIT = 2

    # 관절 PD 게인. _setup_dof_properties / lock_joint / unlock_joint 세 곳에서
    # 각자 리터럴로 중복 정의되어 있던 것을 한 곳으로 모았다.
    # 여기 값만 바꾸면 세 메서드 모두에 일관되게 반영된다.
    _DEFAULT_STIFFNESS = 400.0
    _DEFAULT_DAMPING = 40.0
    _DEFAULT_EFFORT = 100.0
    _LOCKED_STIFFNESS = 15000.0
    _LOCKED_DAMPING = 800.0
    _LOCKED_EFFORT = 100000.0

    # ==========================================================================
    # 초기화
    # ==========================================================================
    def __init__(self, gym, sim, env, handle, stage_registry=None, joint_speed=1.2,
                 capture_radius_xy=0.5, capture_height=0.6,
                 debug_left_tip_offset=None, debug_back_tip_offset=None):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # 자식(패키지) 관리
        self.children = {}          # handle -> {"actor_idx":..., "local_pos":Vec3, "local_rot":Quat, "orig_filters":[...]}
        self.children_locked = False
        self.root_states = None     # setup_tensors()에서 채워짐

        # 상자 자체 상태
        self.box_actor_idx = None
        # 상자 폭/깊이/높이에 맞춰 호출부에서 지정 가능. 기본값은 표준 크기 골판지 상자 기준이며,
        # 에셋 크기가 다른 상자를 쓸 경우 반드시 이 값을 함께 조정해야 scan_and_capture가 정확히 동작한다.
        self.capture_radius_xy = capture_radius_xy   # 상자 폭/깊이 절반 정도
        self.capture_height = capture_height          # 상자 높이 범위

        # 디버그 시각화용 접합부 오프셋 (URDF 치수에 종속되므로 다른 크기 상자는 반드시 재지정)
        self.debug_left_tip_offset = debug_left_tip_offset or gymapi.Vec3(-0.505, 0.0, 0.0)
        self.debug_back_tip_offset = debug_back_tip_offset or gymapi.Vec3(0.0, 0.3, 0.0)

        # 팔레트(platform) 부착 상태
        self.platform_locked = False
        self.platform_actor_idx = None
        self.platform_local_pos = None
        self.platform_local_rot = None

        # 상자의 접기 스테이지 관리
        self.stage_registry = stage_registry
        self.current_stage_name = None

        # 관절 회전 속도
        self.joint_speed = joint_speed

        # 상자 자신의 rigid shape에 공유 필터 비트를 적용
        # (모든 상자 및 자식 패키지가 이 값을 공유하므로 서로 충돌하지 않음)
        self.own_filter_bit = CardboardBoxManager._SHARED_BOX_FILTER_BIT
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.handle)
        for sp in shape_props:
            sp.filter = self.own_filter_bit
        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, shape_props)

        # --- DOF(관절) 설정 및 매핑 ---
        self._setup_dof_properties()
        self.dof_dict = self.gym.get_actor_dof_dict(self.env, self.handle)
        self.dof_names = list(self.dof_dict.keys())
        # print(f"📦 발견된 박스 관절 목록: {self.dof_names}")

        # --- 실시간 관절 제어용 버퍼 ---
        self.num_dofs = len(self.dof_names)
        self.targets = np.zeros(self.num_dofs, dtype=np.float32)
        self.current_targets = np.zeros(self.num_dofs, dtype=np.float32)  # 속도 보간용 현재 타겟

        # --- 시각화용 링크 목록 ---
        self.box_link_names = [
            "link_right", "link_right_up", "link_right_down",
            "link_front", "link_front_up", "link_front_down",
            "link_left",  "link_left_up",  "link_left_down",
            "link_back",  "link_back_up",  "link_back_down"
        ]

        # 실시간 디버그 모니터링을 위한 접합부 글로벌 좌표
        self.left_tip_pos = gymapi.Vec3(0, 0, 0)
        self.back_tip_pos = gymapi.Vec3(0, 0, 0)

        # 링크 이름 -> 액터 내 로컬 인덱스 매핑 (텐서 API 대신 안전한 조회 API 사용)
        self.link_dict = {}
        for name in self.box_link_names:
            idx = self.gym.find_actor_rigid_body_index(self.env, self.handle, name, gymapi.DOMAIN_ACTOR)
            if idx != -1:
                self.link_dict[name] = idx
            else:
                print(f"[경고] 박스 링크 '{name}'을 URDF에서 찾을 수 없습니다.")

    # ==========================================================================
    # DOF(관절) 기본 설정
    # ==========================================================================
    def _setup_dof_properties(self):
        """모든 관절을 위치 제어(PD) 모드로 기본값 설정."""
        box_dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        for i in range(len(box_dof_props)):
            box_dof_props[i]['driveMode'] = gymapi.DOF_MODE_POS
            box_dof_props[i]['stiffness'] = CardboardBoxManager._DEFAULT_STIFFNESS
            box_dof_props[i]['damping'] = CardboardBoxManager._DEFAULT_DAMPING
            box_dof_props[i]['effort'] = CardboardBoxManager._DEFAULT_EFFORT
        self.gym.set_actor_dof_properties(self.env, self.handle, box_dof_props)

    def _set_joint_target(self, joint_name, target_angle):
        """이름으로 특정 관절의 목표 각도를 설정."""
        if joint_name in self.dof_dict:
            idx = self.dof_dict[joint_name]
            self.targets[idx] = target_angle

    # ==========================================================================
    # 등록된 스테이지 적용
    # ==========================================================================
    def apply_stage(self, stage_name):
        """레지스트리에 등록된 스테이지의 관절 목표를 self.targets에 반영."""
        if self.stage_registry is None:
            raise RuntimeError("stage_registry가 주입되지 않았습니다. 생성자에 전달하세요.")

        stage = self.stage_registry.get(stage_name)
        for joint_name, angle_deg in stage["joint_targets"].items():
            self._set_joint_target(joint_name, np.radians(angle_deg))

        self.current_stage_name = stage_name
        return stage

    def _force_spawn_stage_by_name(self, stage_name):
        """
        지정한 스테이지를 적용한 뒤, 물리 엔진 상태를 강제로 동기화하고
        해당 스테이지에 등록된 lock_joints를 잠근다. (첫 프레임 튐 방지)
        """
        stage = self.apply_stage(stage_name)

        self.current_targets = np.copy(self.targets)

        dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
        dof_states['pos'] = self.targets
        dof_states['vel'] = 0.0
        self.gym.set_actor_dof_states(self.env, self.handle, dof_states, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.targets)

        for joint_name in stage["lock_joints"]:
            self.lock_joint(joint_name)

    def set_initial_spawn_stage(self, stage_name):
        """등록된 이름의 스테이지 상태로 초기 스폰 강제 설정."""
        self._force_spawn_stage_by_name(stage_name)

    # ==========================================================================
    # 매 프레임 관절 업데이트
    # ==========================================================================
    def update_joints(self, dt=1.0 / 30.0):
        """목표 각도(self.targets)를 향해 속도 제한을 두고 부드럽게 이동시킨 뒤 물리 엔진에 반영."""
        if self.num_dofs == 0:
            return

        controlled_targets = np.copy(self.targets)

        # 최종 속도 제한 필터 통과 (연속적인 접히는 모션 유지)
        max_step = self.joint_speed * dt
        diff = controlled_targets - self.current_targets
        self.current_targets += np.clip(diff, -max_step, max_step)

        # 물리 인스턴스에 타겟 주입
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.current_targets)

    # ==========================================================================
    # 관절 속도 실행 중 변경
    # ==========================================================================
    def set_joint_speed(self, new_speed):
        """
        관절 접기/펴기 속도(rad/s)를 실행 중에 변경한다.
        update_joints()가 매 프레임 self.joint_speed를 참조해 속도 제한 보간을 하므로,
        이 값을 바꾸면 별도 재시작 없이 다음 프레임부터 바로 반영된다.
        (현재 진행 중인 current_targets → targets 보간의 남은 거리는 유지된 채
         이후 스텝 폭만 새 속도로 바뀐다.)
        """
        if new_speed <= 0:
            raise ValueError(f"joint_speed는 0보다 커야 합니다: {new_speed}")
        self.joint_speed = float(new_speed)

    def get_joint_speed(self):
        """현재 설정된 관절 속도(rad/s)를 반환."""
        return self.joint_speed

    # ==========================================================================
    # 관절 잠금 / 해제
    # ==========================================================================
    def lock_joint(self, joint_name):
        """외력 및 충돌에 밀리지 않도록 가동 범위를 꽉 잠그고 최대 토크를 부여."""
        if joint_name not in self.dof_dict:
            print(f"[경고] 잠그려는 관절 '{joint_name}'을 찾을 수 없습니다.")
            return

        idx = self.dof_dict[joint_name]
        box_dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        lock_angle = self.targets[idx]

        # 상하한선을 오차 없이 고정
        box_dof_props[idx]['lower'] = lock_angle
        box_dof_props[idx]['upper'] = lock_angle

        # 제어 모드는 위치 제어 유지
        box_dof_props[idx]['driveMode'] = gymapi.DOF_MODE_POS

        # 강성/감쇠/최대 토크를 크게 상향하여 외력에 밀리지 않게 함
        box_dof_props[idx]['stiffness'] = CardboardBoxManager._LOCKED_STIFFNESS
        box_dof_props[idx]['damping'] = CardboardBoxManager._LOCKED_DAMPING
        box_dof_props[idx]['effort'] = CardboardBoxManager._LOCKED_EFFORT

        self.gym.set_actor_dof_properties(self.env, self.handle, box_dof_props)

        # 실시간 타겟 버퍼도 잠금 각도로 강제 동기화 (오차 누적 방지)
        self.current_targets[idx] = lock_angle
        # print(f"🔒 관절 절대 잠금 완료: {joint_name} -> {np.degrees(lock_angle):.1f}도 "
        #       f"(Effort: {box_dof_props[idx]['effort']})")

    def unlock_joint(self, joint_name):
        """새로운 모션을 위해 기본 속성으로 안전하게 리셋."""
        if joint_name not in self.dof_dict:
            return

        idx = self.dof_dict[joint_name]
        box_dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        box_dof_props[idx]['lower'] = -(np.pi * 2)
        box_dof_props[idx]['upper'] = (np.pi * 2)
        box_dof_props[idx]['stiffness'] = CardboardBoxManager._DEFAULT_STIFFNESS
        box_dof_props[idx]['damping'] = CardboardBoxManager._DEFAULT_DAMPING
        box_dof_props[idx]['effort'] = CardboardBoxManager._DEFAULT_EFFORT

        self.gym.set_actor_dof_properties(self.env, self.handle, box_dof_props)
        # print(f"🔓 관절 잠금 해제: {joint_name}")

    # ==========================================================================
    # 디버그 시각화
    # ==========================================================================
    def draw_debug_visuals(self, viewer, sphere_radius=0.025, color=(1.0, 0.0, 0.0)):
        """
        텐서 API 없이 상자 액터의 실시간 리지드 바디 상태를 직접 파싱하여
        상자 표면 및 주요 접합부에 디버그 구체를 그린다.
        """
        if self.num_dofs == 0:
            return

        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)
        if body_states is None or len(body_states) == 0:
            return

        # 1. 모든 링크 중심점에 기본 구체(Red) 그리기
        for link_name in self.box_link_names:
            if link_name in self.link_dict:
                actor_local_idx = self.link_dict[link_name]
                pose = body_states['pose'][actor_local_idx]
                origin = gymapi.Vec3(pose['p']['x'], pose['p']['y'], pose['p']['z'])

                sphere_geom = gymutil.WireframeSphereGeometry(sphere_radius, 10, 10, gymapi.Transform(p=origin), color)
                gymutil.draw_lines(sphere_geom, self.gym, viewer, self.env, gymapi.Transform())

        # 2. 🟢 link_left 접합 기준점
        if "link_left" in self.link_dict:
            idx_left = self.link_dict["link_left"]
            pose_left = body_states['pose'][idx_left]

            p_left = pose_left['p']
            r_left = pose_left['r']
            q_left = gymapi.Quat(r_left['x'], r_left['y'], r_left['z'], r_left['w'])

            local_offset_left = self.debug_left_tip_offset
            rotated_offset = q_left.rotate(local_offset_left)

            self.left_tip_pos = gymapi.Vec3(
                p_left['x'] + rotated_offset.x,
                p_left['y'] + rotated_offset.y,
                p_left['z'] + rotated_offset.z
            )

            left_geom = gymutil.WireframeSphereGeometry(
                sphere_radius * 1.4, 10, 10, gymapi.Transform(p=self.left_tip_pos), (0.0, 1.0, 0.0)
            )
            gymutil.draw_lines(left_geom, self.gym, viewer, self.env, gymapi.Transform())

        # 3. 🔵 link_back 접합 기준점
        if "link_back" in self.link_dict:
            idx_back = self.link_dict["link_back"]
            pose_back = body_states['pose'][idx_back]

            p_back = pose_back['p']
            r_back = pose_back['r']
            q_back = gymapi.Quat(r_back['x'], r_back['y'], r_back['z'], r_back['w'])

            local_offset_back = self.debug_back_tip_offset
            rotated_offset_back = q_back.rotate(local_offset_back)

            self.back_tip_pos = gymapi.Vec3(
                p_back['x'] + rotated_offset_back.x,
                p_back['y'] + rotated_offset_back.y,
                p_back['z'] + rotated_offset_back.z
            )

            back_geom = gymutil.WireframeSphereGeometry(
                sphere_radius * 1.4, 10, 10, gymapi.Transform(p=self.back_tip_pos), (0.0, 0.0, 1.0)
            )
            gymutil.draw_lines(back_geom, self.gym, viewer, self.env, gymapi.Transform())

    # ==========================================================================
    # 텐서 초기화 (prepare_sim 이후)
    # ==========================================================================
    def setup_tensors(self):
        """gym.prepare_sim(sim) 이후 호출."""
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root_tensor).view(-1, 13)
        self.box_actor_idx = self.gym.get_actor_index(self.env, self.handle, gymapi.DOMAIN_SIM)

    # ==========================================================================
    # 자식(패키지) 관리
    # ==========================================================================
    def register_child(self, actor_handle):
        """상자 안에 들어갈 패키지를 자식으로 등록 (오프셋은 아직 계산 안 함)."""
        # [수정] 중복 등록 가드: 이미 이 상자의 자식으로 등록돼 있으면 아무 것도 하지 않는다.
        # 재등록을 허용하면 이미 공유 필터로 바뀐 값이 orig_filters에 다시 저장되어
        # 이후 unregister_child에서 원래의 '진짜 최초' 필터 값으로 복원되지 않는 버그가 생긴다.
        if actor_handle in self.children:
            print(f"[경고] 이미 등록된 자식을 다시 register_child 시도: {actor_handle} (무시)")
            return

        actor_idx = self.gym.get_actor_index(self.env, actor_handle, gymapi.DOMAIN_SIM)

        # 이 상자가 속한 공유 콜리전 그룹에 자식 패키지도 편입
        # (모든 상자·모든 자식 패키지가 같은 필터 비트를 쓰므로 서로 충돌하지 않음)
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, actor_handle)
        # 나중에 unlock_children/unregister_child에서 원래 값으로 복원할 수 있도록
        # 편입 전 필터 값을 저장해둔다.
        orig_filters = [sp.filter for sp in shape_props]
        for sp in shape_props:
            sp.filter = self.own_filter_bit
        self.gym.set_actor_rigid_shape_properties(self.env, actor_handle, shape_props)

        self.children[actor_handle] = {
            "actor_idx": actor_idx,
            "local_pos": None,
            "local_rot": None,
            "orig_filters": orig_filters,
        }

    def unregister_child(self, actor_handle):
        """
        자식 하나를 편입 해제하고 콜리전 필터를 register_child 이전 값으로 복원.
        상자 소속에서 완전히 벗어나 다시 독립된 물리 오브젝트로 되돌릴 때 사용.
        """
        if actor_handle not in self.children:
            return

        st = self.children.pop(actor_handle)
        orig_filters = st.get("orig_filters")
        if orig_filters:
            shape_props = self.gym.get_actor_rigid_shape_properties(self.env, actor_handle)
            for sp, orig in zip(shape_props, orig_filters):
                sp.filter = orig
            self.gym.set_actor_rigid_shape_properties(self.env, actor_handle, shape_props)
        # print(f"📦 패키지 편입 해제: {actor_handle}")

    def lock_children_offsets(self):
        """
        현재 시점(패키지들이 상자 안에 안착한 뒤)의 상대 위치/회전을 고정.
        이후 update_children()이 이 오프셋을 기준으로 계속 따라붙게 만든다.
        상자를 옮기기 직전(예: stage_3_close_top 호출 시점)에 한 번만 호출.
        """
        box_pos = self.root_states[self.box_actor_idx, 0:3]
        box_rot = self.root_states[self.box_actor_idx, 3:7]
        box_q = gymapi.Quat(box_rot[0].item(), box_rot[1].item(), box_rot[2].item(), box_rot[3].item())
        box_q_inv = box_q.inverse()
        box_p = gymapi.Vec3(box_pos[0].item(), box_pos[1].item(), box_pos[2].item())

        for handle, st in self.children.items():
            c_pos = self.root_states[st["actor_idx"], 0:3]
            c_rot = self.root_states[st["actor_idx"], 3:7]
            c_q = gymapi.Quat(c_rot[0].item(), c_rot[1].item(), c_rot[2].item(), c_rot[3].item())

            delta_world = gymapi.Vec3(
                c_pos[0].item() - box_p.x, c_pos[1].item() - box_p.y, c_pos[2].item() - box_p.z
            )
            st["local_pos"] = box_q_inv.rotate(delta_world)
            st["local_rot"] = box_q_inv * c_q

        self.children_locked = True
        # print(f"📦 상자 자식 {len(self.children)}개 오프셋 고정 완료")

    def unlock_children(self):
        """
        상자를 열거나 패키지를 다시 자유롭게 둘 때(하역) 호출.
        등록된 모든 자식을 편입 해제하고 콜리전 필터를 원래 값으로 복원해,
        이후 순수 물리 시뮬레이션(중력/충돌)에 완전히 맡긴다.
        이 호출 한 번으로 하역이 끝나므로, scan_and_capture를 매 프레임
        돌리며 이탈을 감지할 필요가 없다.
        """
        for handle in list(self.children.keys()):
            self.unregister_child(handle)
        self.children_locked = False

    def update_children(self):
        """
        매 프레임 호출. 상자가 어떻게 옮겨지든(컨베이어, 로봇팔, 지게차 등)
        고정된 오프셋 기준으로 자식들을 완전 강체처럼 순간 스냅(즉시 따라붙음)시킨다.
        """
        if not self.children_locked or not self.children:
            return

        box_pos = self.root_states[self.box_actor_idx, 0:3]
        box_rot = self.root_states[self.box_actor_idx, 3:7]
        box_q = gymapi.Quat(box_rot[0].item(), box_rot[1].item(), box_rot[2].item(), box_rot[3].item())
        box_p = gymapi.Vec3(box_pos[0].item(), box_pos[1].item(), box_pos[2].item())

        updated_indices = []
        for handle, st in self.children.items():
            target_offset = box_q.rotate(st["local_pos"])
            target_pos = gymapi.Vec3(box_p.x + target_offset.x, box_p.y + target_offset.y, box_p.z + target_offset.z)
            target_rot = box_q * st["local_rot"]

            new_state = self.root_states[st["actor_idx"]].clone()
            new_state[0], new_state[1], new_state[2] = target_pos.x, target_pos.y, target_pos.z
            new_state[3], new_state[4], new_state[5], new_state[6] = target_rot.x, target_rot.y, target_rot.z, target_rot.w
            new_state[7:13] = 0.0   # 텔레포트 방식이라 관성 불필요

            self.root_states[st["actor_idx"]] = new_state
            updated_indices.append(st["actor_idx"])

        if updated_indices:
            idx_t = torch.tensor(updated_indices, dtype=torch.int32, device=self.root_states.device)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(idx_t), len(updated_indices)
            )

    def scan_and_capture(self, candidate_handles, root_states_getter):
        """
        candidate_handles: 씬에 존재하는 모든 패키지 handle 리스트(전역 후보 풀)
        root_states_getter: (actor_handle) -> (x, y, z) 월드 좌표를 반환하는 콜백

        박스 root 위치를 중심으로 xy 반경/높이 범위 안에 있는 후보를 children으로 편입.
        이미 등록된 children은 건드리지 않고, 새로 들어온 것만 추가.
        (이탈 처리는 하지 않는다 — 하역 시에는 unlock_children()이 등록된
        모든 자식을 일괄 편입 해제하므로, 프레임별 이탈 감지가 필요 없다.)
        """
        box_pos = self.root_states[self.box_actor_idx, 0:3]
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()

        for handle in candidate_handles:
            if handle in self.children:
                continue  # 이미 이 상자 소속

            px, py, pz = root_states_getter(handle)
            dx, dy, dz = px - bx, py - by, pz - bz
            dist_xy = (dx ** 2 + dy ** 2) ** 0.5

            if dist_xy < self.capture_radius_xy and abs(dz) < self.capture_height:
                self.register_child(handle)
                # print(f"📦 패키지 자동 편입: {handle} (거리 xy={dist_xy:.3f}, dz={dz:.3f})")

    # ==========================================================================
    # 플랫폼(팔레트) 부착
    # ==========================================================================
    def attach_to_platform(self, platform_actor_handle):
        """
        박스 자신을 platform(팔레트 등)에 오프셋 고정.
        이후 update_platform_lock()이 매 프레임 이 오프셋을 유지시켜,
        platform이 어떻게 움직이든(병진/회전) 박스가 마찰 없이 정확히 따라붙는다.
        """
        self.platform_actor_idx = self.gym.get_actor_index(self.env, platform_actor_handle, gymapi.DOMAIN_SIM)

        plat_pos = self.root_states[self.platform_actor_idx, 0:3]
        plat_rot = self.root_states[self.platform_actor_idx, 3:7]
        plat_q = gymapi.Quat(plat_rot[0].item(), plat_rot[1].item(), plat_rot[2].item(), plat_rot[3].item())
        plat_q_inv = plat_q.inverse()
        plat_p = gymapi.Vec3(plat_pos[0].item(), plat_pos[1].item(), plat_pos[2].item())

        box_pos = self.root_states[self.box_actor_idx, 0:3]
        box_rot = self.root_states[self.box_actor_idx, 3:7]
        box_q = gymapi.Quat(box_rot[0].item(), box_rot[1].item(), box_rot[2].item(), box_rot[3].item())

        delta_world = gymapi.Vec3(
            box_pos[0].item() - plat_p.x, box_pos[1].item() - plat_p.y, box_pos[2].item() - plat_p.z
        )
        self.platform_local_pos = plat_q_inv.rotate(delta_world)
        self.platform_local_rot = plat_q_inv * box_q
        self.platform_locked = True
        # print(f"📦 box_manager 팔레트 고정 완료")

    def detach_from_platform(self):
        self.platform_locked = False
        self.platform_actor_idx = None

    def update_platform_lock(self):
        """매 프레임 호출. platform이 병진/회전 어떻게 하든 박스가 정확히 따라감."""
        if not self.platform_locked:
            return

        plat_pos = self.root_states[self.platform_actor_idx, 0:3]
        plat_rot = self.root_states[self.platform_actor_idx, 3:7]
        plat_q = gymapi.Quat(plat_rot[0].item(), plat_rot[1].item(), plat_rot[2].item(), plat_rot[3].item())
        plat_p = gymapi.Vec3(plat_pos[0].item(), plat_pos[1].item(), plat_pos[2].item())

        target_offset = plat_q.rotate(self.platform_local_pos)
        target_pos = gymapi.Vec3(plat_p.x + target_offset.x, plat_p.y + target_offset.y, plat_p.z + target_offset.z)
        target_rot = plat_q * self.platform_local_rot

        new_state = self.root_states[self.box_actor_idx].clone()
        new_state[0], new_state[1], new_state[2] = target_pos.x, target_pos.y, target_pos.z
        new_state[3], new_state[4], new_state[5], new_state[6] = target_rot.x, target_rot.y, target_rot.z, target_rot.w
        new_state[7:13] = 0.0

        self.root_states[self.box_actor_idx] = new_state
        idx_t = torch.tensor([self.box_actor_idx], dtype=torch.int32, device=self.root_states.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(idx_t), 1
        )