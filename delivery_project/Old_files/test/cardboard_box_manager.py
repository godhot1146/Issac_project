import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch

class CardboardBoxManager:
    _next_filter_bit = 2 

    def __init__(self, gym, sim, env, asset_root, pose=None, fix=True):
        self.gym = gym
        self.sim = sim
        self.env = env

        self.children = {}          # handle -> {"actor_idx":..., "local_pos":Vec3, "local_rot":Quat}
        self.children_locked = False
        self.root_states = None     # setup_tensors에서 채움
        self.box_actor_idx = None
        self.capture_radius_xy = 0.5   # 상자 폭/깊이 절반 정도
        self.capture_height = 0.6      # 상자 높이 범위

        self.platform_locked = False
        self.platform_actor_idx = None
        self.platform_local_pos = None
        self.platform_local_rot = None
        
        # 1. 에셋 및 액터 초기화
        self.opts = gymapi.AssetOptions()
        self.opts.fix_base_link = fix
        self.opts.density = 100.0
        self.asset = self.gym.load_asset(self.sim, asset_root, "urdf/cardboard_box/v2/box_v2.urdf", self.opts)
        
        if pose is None:
            pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 1.5))
        self.handle = self.gym.create_actor(self.env, self.asset, pose, "cordboard_box_asset", -1, 1)
        
        # 🆕 이 위치에 추가
        self.own_filter_bit = CardboardBoxManager._next_filter_bit
        CardboardBoxManager._next_filter_bit <<= 1
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.handle)
        for sp in shape_props:
            sp.filter = self.own_filter_bit
        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, shape_props)

        # 2. DOF(관절) 설정 및 맵핑
        self._setup_dof_properties()
        self.dof_dict = self.gym.get_actor_dof_dict(self.env, self.handle)
        self.dof_names = list(self.dof_dict.keys())
        print(f"📦 발견된 박스 관절 목록: {self.dof_names}")

        # 3. 실시간 관절 제어용 버퍼 배열 및 상태 변수
        self.num_dofs = len(self.dof_names)
        self.targets = np.zeros(self.num_dofs, dtype=np.float32)

        # 실시간 속도 보간 및 닫힌 루프(IK) 상태 제어를 위한 변수
        self.current_targets = np.zeros(self.num_dofs, dtype=np.float32)
        self.current_stage = 0  

        # 시각화용 링크 목록
        self.box_link_names = [
            "link_right", "link_right_up", "link_right_down",
            "link_front", "link_front_up", "link_front_down",
            "link_left",  "link_left_up",  "link_left_down",
            "link_back",  "link_back_up",  "link_back_down"
        ]

        # 🎯 실시간 디버그 모니터링을 위한 접합부 글로벌 좌표 저장 변수
        self.left_tip_pos = gymapi.Vec3(0, 0, 0)
        self.back_tip_pos = gymapi.Vec3(0, 0, 0)

        # 💡 [구조 변경] 에러를 유발하는 Tensor API 버퍼를 완전히 제거하고
        # 액터 내 각 링크의 이름을 순당 인덱스로 매핑하는 가장 안전한 API를 사용합니다.
        self.link_dict = {}
        for name in self.box_link_names:
            idx = self.gym.find_actor_rigid_body_index(self.env, self.handle, name, gymapi.DOMAIN_ACTOR)
            if idx != -1:
                self.link_dict[name] = idx
            else:
                print(f"[경고] 박스 링크 '{name}'을 URDF에서 찾을 수 없습니다.")

        # ==============================================================================
        # 🛠️ [신규 추가] 박스 매니저 내부 수동 제어 전용 상태 변수 초기화
        # ==============================================================================
        self.use_manual = True             # 수동 인터페이스 활성화 플래그 (메인 루프 바인딩용)
        self.manual_mode = False           # 수동 모드 켜짐/꺼짐 상태 토글 변수
        self.selected_dof_idx = 0          # 현재 조작 대상으로 선택된 박스 DOF 인덱스
        self.manual_targets = np.zeros(self.num_dofs, dtype=np.float32) # 수동 제어용 타겟 배열
        # ==============================================================================

        # 상자 밑면이 닫힌 상태로 스폰
        self.current_stage = 0
        # self.stage_0_unfolded_flat()
        # self.lock_joint("joint_right_to_front")
        # self.lock_joint("joint_right_to_back")
        # self.lock_joint("joint_front_to_left")

        #self.set_initial_spawn_stage_fold_down()
        #self.set_initial_spawn_stage_fold_up()

    # ==============================================================================
    # 🛠️ [신규 추가] 외부 키보드 이벤트 연동 핸들러 메서드
    # ==============================================================================
    def fix_base_link(self, fix):
        self.opts.fix_base_link = fix

    def handle_keyboard_event(self, action):
        """메인 루프의 키보드 이벤트 분기 처리 시 호출될 함수"""
        if not self.use_manual:
            return

        # 1. 수동 조작 모드 온/오프 (B 키)
        if action == "toggle_box_manual_mode":
            self.manual_mode = not self.manual_mode
            mode_str = "🔥 상자 수동(MANUAL)" if self.manual_mode else "🤖 상자 자동(AUTOMATIC)"
            print(f"\n[Cardboard Box 제어 모드 변경] 현재 구동 모드: {mode_str}")
            
            if self.manual_mode:
                current_joint_name = self.dof_names[self.selected_dof_idx]
                current_angle = self.targets[self.selected_dof_idx]
                print(f" -> 현재 조작 관절 [{self.selected_dof_idx}]: {current_joint_name} (현재 타겟: {np.degrees(current_angle):.1f}°)")

        # 수동 모드일 때만 하위 키 바인딩 유효화
        if self.manual_mode:
            current_joint_name = self.dof_names[self.selected_dof_idx]

            # 2. 관절(DOF) 순회 선택
            if action == "select_prev_joint":
                self.selected_dof_idx = (self.selected_dof_idx - 1) % self.num_dofs
                new_joint_name = self.dof_names[self.selected_dof_idx]
                print(f"🔄 [상자 관절 선택] ➡️ [{self.selected_dof_idx}]: {new_joint_name} (현재 타겟: {np.degrees(self.targets[self.selected_dof_idx]):.1f}°)")
            elif action == "select_next_joint":
                self.selected_dof_idx = (self.selected_dof_idx + 1) % self.num_dofs
                new_joint_name = self.dof_names[self.selected_dof_idx]
                print(f"🔄 [상자 관절 선택] ➡️ [{self.selected_dof_idx}]: {new_joint_name} (현재 타겟: {np.degrees(self.targets[self.selected_dof_idx]):.1f}°)")
            
            # 3. 관절 각도 미세 조절 제어 (🎯 딕셔너리 거치지 않고 현재 인덱스 버퍼에 직접 가감)
            elif action == "decrease_joint_angle":
                self.targets[self.selected_dof_idx] -= np.radians(5.0)  # 10도 감소
                print(f"📉 [{current_joint_name}] 타겟 변경: {np.degrees(self.targets[self.selected_dof_idx]):.1f}°")

            elif action == "increase_joint_angle":
                self.targets[self.selected_dof_idx] += np.radians(5.0)  # 10도 증가
                print(f"📈 [{current_joint_name}] 타겟 변경: {np.degrees(self.targets[self.selected_dof_idx]):.1f}°")

        # 4. 정보 리포트 출력
        if action == "start_process" and self.manual_mode:
            current_joint_name = self.dof_names[self.selected_dof_idx]
            print(f"\n📊 [Cardboard Box 수동 조작 관절 리포트]")
            print(f" - 선택된 관절 명칭: {current_joint_name} (Index: {self.selected_dof_idx})")
            print(f" - 목적 각도: {self.targets[self.selected_dof_idx]:.4f} rad ({np.degrees(self.targets[self.selected_dof_idx]):.1f}°)")

    def _setup_dof_properties(self):
        box_dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        for i in range(len(box_dof_props)):
            box_dof_props[i]['driveMode'] = gymapi.DOF_MODE_POS
            box_dof_props[i]['stiffness'] = 400.0  
            box_dof_props[i]['damping'] = 40.0     
            box_dof_props[i]['effort'] = 100.0     
        self.gym.set_actor_dof_properties(self.env, self.handle, box_dof_props)

    def _set_joint_target(self, joint_name, target_angle):
        if joint_name in self.dof_dict:
            idx = self.dof_dict[joint_name]
            self.targets[idx] = target_angle

    def stage_0_unfolded_flat(self):
        self._set_joint_target("joint_right_to_front", np.radians(10.0))
        self._set_joint_target("joint_right_to_back",  np.radians(10.0))
        self._set_joint_target("joint_front_to_left",  np.radians(-10.0))
        # self.targets = np.zeros(self.num_dofs, dtype=np.float32)

    def stage_1_fold_sides(self):
        self._set_joint_target("joint_right_to_front", np.radians(90.0))
        self._set_joint_target("joint_right_to_back",  np.radians(90.0))
        self._set_joint_target("joint_front_to_left",  np.radians(-90.0))

    def stage_2_fold_back_to_down1(self):
        self._set_joint_target("joint_back_to_down",  np.radians(90.0))
        self._set_joint_target("joint_front_to_down", np.radians(-90.0))

    def stage_2_fold_back_to_down2(self):
        self._set_joint_target("joint_back_to_down",  np.radians(90.0))
        self._set_joint_target("joint_front_to_down", np.radians(-90.0))

    def stage_2_fold_back_to_down3(self):
        self._set_joint_target("joint_back_to_down",  np.radians(90.0))
        self._set_joint_target("joint_front_to_down", np.radians(-90.0))
    
    def stage_2_fold_back_to_down4(self):
        self._set_joint_target("joint_back_to_down",  np.radians(90.0))
        self._set_joint_target("joint_front_to_down", np.radians(-90.0))

    def close_side_bottom(self):
        self._set_joint_target("joint_right_to_down", np.radians(-50.0))
        self._set_joint_target("joint_left_to_down",  np.radians(50.0))

    def close_side_bottom2(self):
        self._set_joint_target("joint_right_to_down", np.radians(-90.0))
        self._set_joint_target("joint_left_to_down",  np.radians(90.0))

    def stage_2_close_bottom(self):
        self.stage_1_fold_sides()
        self._set_joint_target("joint_right_to_down", np.radians(-90.0))
        self._set_joint_target("joint_left_to_down",  np.radians(90.0))
        self._set_joint_target("joint_front_to_down", np.radians(-90.0))
        self._set_joint_target("joint_back_to_down",  np.radians(90.0))

    def stage_3_close_top(self):
        self.stage_2_close_bottom()
        self._set_joint_target("joint_right_to_up", np.radians(90.0))
        self._set_joint_target("joint_left_to_up",  np.radians(-90.0))
        self._set_joint_target("joint_front_to_up", np.radians(90.0))
        self._set_joint_target("joint_back_to_up",  np.radians(-90.0))

    def set_initial_spawn_stage_fold_down(self):
        """
        [신규 함수] 상자가 처음 스폰될 때 펼쳐진 상태가 아닌, 
        특정 스테이지 상태로 물리 엔진에 강제 고정하여 시작하도록 만듭니다.
        """
        # 1. 원하는 초기 스폰 상태 함수 호출 (여기서 바닥 접힘 상태 주입)
        self.stage_2_close_bottom()
        
        # 2. 속도 제한 필터(update_joints) 우회를 위해 현재 타겟 버퍼도 동기화
        self.current_targets = np.copy(self.targets)
        
        # 3. Isaac Gym 물리 엔진 내 실제 관절 상태(State) 배열 생성 및 강제 주입
        #    (첫 프레임에 튀는 현상을 막기 위해 초기 관절 위치 자체를 변형)
        dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
        dof_states['pos'] = self.targets  # 목표 각도를 초기 위치로 설정
        dof_states['vel'] = 0.0           # 초기 속도는 0
        
        # 월드에 반영
        self.gym.set_actor_dof_states(self.env, self.handle, dof_states, gymapi.STATE_ALL)
        
        # 4. PD 제어기의 최초 목표 포지션(Target)도 동일하게 일치시킴
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.targets)

        self.lock_joint("joint_right_to_down")
        self.lock_joint("joint_left_to_down")
        self.lock_joint("joint_front_to_down")
        self.lock_joint("joint_back_to_down")
        
        print(f"🚀 상자의 초기 스폰 상태가 성공적으로 설정되었습니다. (현재 Stage: {self.current_stage})")

    def set_initial_spawn_stage_fold_up(self):
        """
        [신규 함수] 상자가 처음 스폰될 때 펼쳐진 상태가 아닌, 
        특정 스테이지 상태로 물리 엔진에 강제 고정하여 시작하도록 만듭니다.
        """
        # 1. 원하는 초기 스폰 상태 함수 호출 (여기서 바닥 접힘 상태 주입)
        self.stage_3_close_top()
        
        # 2. 속도 제한 필터(update_joints) 우회를 위해 현재 타겟 버퍼도 동기화
        self.current_targets = np.copy(self.targets)
        
        # 3. Isaac Gym 물리 엔진 내 실제 관절 상태(State) 배열 생성 및 강제 주입
        #    (첫 프레임에 튀는 현상을 막기 위해 초기 관절 위치 자체를 변형)
        dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
        dof_states['pos'] = self.targets  # 목표 각도를 초기 위치로 설정
        dof_states['vel'] = 0.0           # 초기 속도는 0
        
        # 월드에 반영
        self.gym.set_actor_dof_states(self.env, self.handle, dof_states, gymapi.STATE_ALL)
        
        # 4. PD 제어기의 최초 목표 포지션(Target)도 동일하게 일치시킴
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.targets)

        self.lock_joint("joint_right_to_up")
        self.lock_joint("joint_left_to_up")
        self.lock_joint("joint_front_to_up")
        self.lock_joint("joint_back_to_up")
        
        print(f"🚀 상자의 초기 스폰 상태가 성공적으로 설정되었습니다. (현재 Stage: {self.current_stage})")

    def update_joints(self, dt=1.0/30.0):
        if self.num_dofs == 0:
            return

        controlled_targets = np.copy(self.targets)

        # 1. 최종 속도 제한 필터 통과
        max_step = 1.2 * dt
        diff = controlled_targets - self.current_targets
        self.current_targets += np.clip(diff, -max_step, max_step)

        # 2. 물리 인스턴스 타겟 주입 (물리 엔진이 simulate할 때 반영됨)
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.current_targets)


    def draw_debug_visuals(self, viewer, sphere_radius=0.025, color=(1.0, 0.0, 0.0)):
        """
        [완전 재작성] 텐서 충돌 없이 상자 액터의 현재 실시간 리지드 바디 상태를 
        다이렉트로 파싱하여 상자 표면에 정확히 구체를 그려냅니다.
        """
        if self.num_dofs == 0:
            return

        # 💡 텐서 대신 이 액터의 순수한 리지드 바디 상태 배열을 한 번에 가져옵니다.
        # 이 데이터는 물리 엔진 연산이 완료된 즉시 월드 좌표계 기준으로 동기화됩니다.
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)
        if body_states is None or len(body_states) == 0:
            return

        # 1. 상자의 모든 링크 중심점에 기본 구체(Red) 그리기
        for link_name in self.box_link_names:
            if link_name in self.link_dict:
                actor_local_idx = self.link_dict[link_name]
                pose = body_states['pose'][actor_local_idx]
                
                # 월드 좌표 추출
                origin = gymapi.Vec3(pose['p']['x'], pose['p']['y'], pose['p']['z'])
                
                sphere_geom = gymutil.WireframeSphereGeometry(sphere_radius, 10, 10, gymapi.Transform(p=origin), color)
                gymutil.draw_lines(sphere_geom, self.gym, viewer, self.env, gymapi.Transform())

        # ==============================================================================
        # 🎯 [모든 관절 규칙화] 피드백 분석 기반 규칙형 순환 매핑 알고리즘
        # ==============================================================================
        if getattr(self, 'manual_mode', False):
            # URDF 순서와 실제 구동 인덱스의 엇갈림 체인을 완벽하게 동기화하기 위한 전체 매핑 규칙
            if not hasattr(self, 'global_urdf_link_map'):
                # 각 관절이 원래 통제해야 하는 명목상의 자식 링크 정의
                nominal_joint_to_link = {
                    "joint_right_to_up":    "link_right_up",
                    "joint_right_to_down":  "link_right_down",
                    "joint_right_to_front": "link_front",
                    "joint_front_to_up":    "link_front_up",
                    "joint_front_to_down":  "link_front_down",
                    "joint_front_to_left":  "link_left",
                    "joint_left_to_up":     "link_left_up",
                    "joint_left_to_down":   "link_left_down",
                    "joint_right_to_back":  "link_back",
                    "joint_back_to_up":     "link_back_up",
                    "joint_back_to_down":   "link_back_down"
                }

                # 사용자님의 피드백 패턴(A 선택시 B 작동) 규칙을 수학적/구조적 순환 체인으로 생성
                self.global_urdf_link_map = {}
                
                # 제보된 5개 핵심 관절 순환 루프 강제 주입
                self.global_urdf_link_map["joint_back_to_down"]  = nominal_joint_to_link["joint_right_to_back"]
                self.global_urdf_link_map["joint_right_to_back"] = nominal_joint_to_link["joint_left_to_down"]
                self.global_urdf_link_map["joint_left_to_down"]  = nominal_joint_to_link["joint_front_to_down"]
                self.global_urdf_link_map["joint_front_to_down"] = nominal_joint_to_link["joint_back_to_up"]
                self.global_urdf_link_map["joint_back_to_up"]    = nominal_joint_to_link["joint_back_to_down"]

                # 나머지 관절들도 동일한 오프셋 패턴(이전 관절 매핑) 규칙에 따라 자동 확장 매핑
                leftover_joints = [j for j in self.dof_names if j not in self.global_urdf_link_map]
                for j_name in leftover_joints:
                    # 현재 관절의 리스트 상 인덱스 탐색
                    idx = self.dof_names.index(j_name)
                    # 규칙: 실제 움직이는 죠인트는 리스트 상 바로 직전(idx - 1) 죠인트의 물리 바디
                    actual_moving_joint = self.dof_names[(idx - 1) % self.num_dofs]
                    self.global_urdf_link_map[j_name] = nominal_joint_to_link.get(actual_moving_joint, nominal_joint_to_link[j_name])

            # 1. 제어 액션 창에 떠 있는 현재 관절 이름
            current_joint_name = self.dof_names[self.selected_dof_idx]
            
            # 2. 통합 규칙 테이블에서 매칭되는 자식 링크 이름 찾기
            target_link_name = self.global_urdf_link_map.get(current_joint_name, None)

            # 3. 인덱스 매칭 후 화면에 하얀 구체 가시화
            if target_link_name and target_link_name in self.link_dict:
                target_link_idx = self.link_dict[target_link_name]
                
                if target_link_idx < len(body_states['pose']):
                    pose_manual = body_states['pose'][target_link_idx]
                    origin_manual = gymapi.Vec3(pose_manual['p']['x'], pose_manual['p']['y'], pose_manual['p']['z'])
                    
                    manual_geom = gymutil.WireframeSphereGeometry(sphere_radius * 1.8, 14, 14, gymapi.Transform(p=origin_manual), (1.0, 1.0, 1.0))
                    gymutil.draw_lines(manual_geom, self.gym, viewer, self.env, gymapi.Transform())
        # ==============================================================================
                                
        # 2. 🟢 link_left 접합 기준점 연산 및 시각화 (Green)
        if "link_left" in self.link_dict:
            idx_left = self.link_dict["link_left"]
            pose_left = body_states['pose'][idx_left]
            
            p_left = pose_left['p']
            r_left = pose_left['r']
            q_left = gymapi.Quat(r_left['x'], r_left['y'], r_left['z'], r_left['w'])
            
            local_offset_left = gymapi.Vec3(-0.505, 0.0, 0.0)
            rotated_offset = q_left.rotate(local_offset_left)
            
            self.left_tip_pos = gymapi.Vec3(
                p_left['x'] + rotated_offset.x,
                p_left['y'] + rotated_offset.y,
                p_left['z'] + rotated_offset.z
            )
            
            left_geom = gymutil.WireframeSphereGeometry(sphere_radius * 1.4, 10, 10, gymapi.Transform(p=self.left_tip_pos), (0.0, 1.0, 0.0))
            gymutil.draw_lines(left_geom, self.gym, viewer, self.env, gymapi.Transform())

        # 3. 🔵 link_back 접합 기준점 연산 및 시각화 (Blue)
        if "link_back" in self.link_dict:
            idx_back = self.link_dict["link_back"]
            pose_back = body_states['pose'][idx_back]
            
            p_back = pose_back['p']
            r_back = pose_back['r']
            q_back = gymapi.Quat(r_back['x'], r_back['y'], r_back['z'], r_back['w'])
            
            local_offset_back = gymapi.Vec3(0.0, 0.3, 0.0)
            rotated_offset_back = q_back.rotate(local_offset_back)
            
            self.back_tip_pos = gymapi.Vec3(
                p_back['x'] + rotated_offset_back.x,
                p_back['y'] + rotated_offset_back.y,
                p_back['z'] + rotated_offset_back.z
            )

            back_geom = gymutil.WireframeSphereGeometry(sphere_radius * 1.4, 10, 10, gymapi.Transform(p=self.back_tip_pos), (0.0, 0.0, 1.0))
            gymutil.draw_lines(back_geom, self.gym, viewer, self.env, gymapi.Transform())

    def update_joints(self, dt=1.0/30.0):
        if self.num_dofs == 0:
            return

        controlled_targets = np.copy(self.targets)

        # 1. 최종 속도 제한 필터 통과 (연속적인 접히는 모션 유지)
        max_step = 1.2 * dt
        diff = controlled_targets - self.current_targets
        self.current_targets += np.clip(diff, -max_step, max_step)

        # 2. 물리 인스턴스 타겟 주입
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.current_targets)

    def lock_joint(self, joint_name):
        """[개선안] 외력 및 충돌에 밀리지 않도록 가동 범위를 꽉 잠그고 최대 토크를 부여합니다."""
        if joint_name not in self.dof_dict:
            print(f"[경고] 잠그려는 관절 '{joint_name}'을 찾을 수 없습니다.")
            return

        idx = self.dof_dict[joint_name]
        box_dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        
        # 현재 이 관절이 도달해야 했던 목표 각도
        lock_angle = self.targets[idx]
        
        # 1. 상하한선을 오차 없이 고정
        box_dof_props[idx]['lower'] = lock_angle
        box_dof_props[idx]['upper'] = lock_angle
        
        # 2. 제어 모드는 위치 제어 유지
        box_dof_props[idx]['driveMode'] = gymapi.DOF_MODE_POS
        
        # 3. 🔥 핵심: 강성과 감쇠비를 극대화하고, 최대 토크(effort) 한계를 해제 수준으로 상향
        box_dof_props[idx]['stiffness'] = 15000.0  # 기존 4000에서 대폭 상향
        box_dof_props[idx]['damping'] = 800.0      # 진동 방지를 위한 댐핑 최적화
        box_dof_props[idx]['effort'] = 100000.0    # 외력에 밀리지 않도록 최대 힘 한계 폭발적 상향
        
        self.gym.set_actor_dof_properties(self.env, self.handle, box_dof_props)
        
        # 4. 실시간 타겟 버퍼도 현재 잠금 각도로 강제 동기화 (오차 누적 방지)
        self.current_targets[idx] = lock_angle
        print(f"🔒 관절 절대 잠금 완료: {joint_name} -> {np.degrees(lock_angle):.1f}도 (Effort: {box_dof_props[idx]['effort']})")

    def unlock_joint(self, joint_name):
        """[개선안] 새로운 모션을 위해 기본 속성으로 안전하게 리셋합니다."""
        if joint_name not in self.dof_dict:
            return

        idx = self.dof_dict[joint_name]
        box_dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        
        # 기본 가동 범위 및 초기 제어 강성/최대 토크로 리셋
        box_dof_props[idx]['lower'] = -np.pi
        box_dof_props[idx]['upper'] = np.pi
        box_dof_props[idx]['stiffness'] = 400.0
        box_dof_props[idx]['damping'] = 40.0
        box_dof_props[idx]['effort'] = 100.0  # 초기값 복원
        
        self.gym.set_actor_dof_properties(self.env, self.handle, box_dof_props)
        print(f"🔓 관절 잠금 해제: {joint_name}")

    def setup_tensors(self):
        """gym.prepare_sim(sim) 이후 호출."""
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root_tensor).view(-1, 13)
        self.box_actor_idx = self.gym.get_actor_index(self.env, self.handle, gymapi.DOMAIN_SIM)

    def register_child(self, actor_handle):
        """상자 안에 들어갈 패키지를 자식으로만 등록 (오프셋은 아직 계산 안 함)."""
        actor_idx = self.gym.get_actor_index(self.env, actor_handle, gymapi.DOMAIN_SIM)
        self.children[actor_handle] = {"actor_idx": actor_idx, "local_pos": None, "local_rot": None}

        # 🆕 자기 상자와만 콜리전 제외
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, actor_handle)
        for sp in shape_props:
            sp.filter = self.own_filter_bit
        self.gym.set_actor_rigid_shape_properties(self.env, actor_handle, shape_props)

    def lock_children_offsets(self):
        """
        현재 시점(패키지들이 상자 안에 안착한 뒤)의 상대 위치/회전을 '고정'.
        이후부터 update_children()이 이 오프셋을 기준으로 계속 따라붙게 만든다.
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
        print(f"📦 상자 자식 {len(self.children)}개 오프셋 고정 완료")

    def unlock_children(self):
        """상자를 열거나 패키지를 다시 자유롭게 둘 때 해제."""
        self.children_locked = False

    def update_children(self, pos_gain=1.0, rot_gain=1.0):
        """
        매 프레임 호출. 상자를 누가 어떻게 옮겼든(컨베이어, 로봇팔, 향후 지게차 등)
        상관없이 고정된 오프셋 기준으로 자식들을 붙여준다.
        pos_gain=1.0이면 완전 강체처럼 순간 스냅(즉시 따라붙음).
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
            new_state[7:13] = 0.0   # 속도는 0으로 (텔레포트 방식이라 관성 불필요)

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
                             (get_actor_world_pos 재사용 가능)

        박스 root 위치를 중심으로 xy 반경/높이 범위 안에 있는 후보를 children으로 편입.
        이미 등록된 children은 건드리지 않고, 새로 들어온 것만 추가.
        """
        box_pos = self.root_states[self.box_actor_idx, 0:3]
        bx, by, bz = box_pos[0].item(), box_pos[1].item(), box_pos[2].item()

        for handle in candidate_handles:
            if handle in self.children:
                continue  # 이미 이 상자 소속

            px, py, pz = root_states_getter(handle)
            dx, dy, dz = px - bx, py - by, pz - bz
            dist_xy = (dx**2 + dy**2) ** 0.5

            if dist_xy < self.capture_radius_xy and abs(dz) < self.capture_height:
                self.register_child(handle)
                print(f"📦 패키지 자동 편입: {handle} (거리 xy={dist_xy:.3f}, dz={dz:.3f})")

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
        print(f"📦 box_manager1 팔레트 고정 완료")

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