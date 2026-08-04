import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch

class FrankaController:
    def __init__(self, gym, sim, env, handle, pump_link_name="cobot_pump", scale=1.6):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle
        self.scale = scale  # 1.6 스케일 반영
        
        # 펌프(엔드이펙터) 인덱스 매핑
        self.pump_index = gym.find_actor_rigid_body_index(env, handle, pump_link_name, gymapi.DOMAIN_ACTOR)
        if self.pump_index == -1:
            print(f"[경고] {pump_link_name} 링크를 찾을 수 없습니다. URDF를 확인하세요.")

        # 동작 시퀀스 상태 변수
        self.state = -1  # 0: ready, 1: pick, 2: close, 3: lift, 4: rotate, 5: place, 6: release, 7: lift_up
        self.step_duration = 150
        self.done = False

        # 관절 상태 정의 (스케일이 바뀌어도 관절 각도(Rad) 자체는 동일하게 유지됨)
        self.states = {
            "straight": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ready":   [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
            "pick":    [0, 0.8, 0, -1.75, 0, 2.50, 0.90],
            "close":   [0, 0.8, 0, -1.75, 0, 2.50, 0.90],
            "lift":    [0, -0.5, 0, -2.0, 0, 1.50, 0.90],
            "rotate":  [1.571, -0.5, 0, -2.0, 0, 2.50, 0.90],
            "place":   [1.571, 0.35, 0, -1.45, 0, 2.50, 0.6],
            "release": [1.571, 0.35, 0, -1.45, 0, 2.50, 0.6],
            "lift_up": [1.571, -0.5, 0, -1.45, 0, 1.50, 0.6]
        }

        # --- [IK 핵심] 한도 가동 범위 조인트 정보 추출 ---
        self.dof_props = gym.get_actor_dof_properties(env, handle)
        self.dof_lower_limits = self.dof_props["lower"]
        self.dof_upper_limits = self.dof_props["upper"]

        self.state = -1  
        self.step_duration = 150
        self.done = False

        # 💡 [IK 변경점] 각도가 아닌 1.6배 확장된 맵 기준 "3D 태스크 공간 좌표" 정의
        # [X, Y, Z, Qx, Qy, Qz, Qw] 형태 (그리퍼가 아래를 바라보도록 셋팅)
        down_quat = [1.0, 0.0, 0.0, 0.0]  # 아래를 향하는 기본 사원수 예시
        
        self.states = {
            "straight":     [0.0,  0.0, 1.6, 1.0, 0.0, 0.0, 0.0], # 시작 수직 정렬
            "move_front":   [0.5,  0.0, 0.5, 1.0, 0.0, 0.0, 0.0], # 1. 앞으로 50cm
            "move_back":    [-0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], # 2. 뒤로 50cm
            "move_left":    [0.0,  0.5, 0.5, 1.0, 0.0, 0.0, 0.0], # 3. 왼쪽으로 50cm
            "move_right":   [0.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0]  # 4. 오른쪽으로 50cm
        }

        # 진공 흡착(Snap) 관련 변수 내부 탑재
        self.is_attached = False
        self.is_snapping = False
        self.snap_frame_count = 0
        self.SNAP_DURATION = 15
        
        self.snap_start_pos = None
        self.snap_start_rot = None

    def solve_ik(self, target_pos, target_rot):
        """ Isaac Gym 정식 야코비안 텐서 API를 이용한 IK 풀이 함수 """
        # 1. 💡 [정식 수정] 오타가 있던 함수명에 's'를 붙여 복수형으로 호출합니다.
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)  # 👈 'tensor' 뒤에 's'를 꼭 붙여야 합니다!
        
        # 현재 엔드이펙터(펌프)의 글로벌 포즈 추출
        rb_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        current_pose = rb_states['pose'][self.pump_index]
        
        current_pos = np.array([current_pose['p']['x'], current_pose['p']['y'], current_pose['p']['z']])
        current_rot = np.array([current_pose['r']['x'], current_pose['r']['y'], current_pose['r']['z'], current_pose['r']['w']])

        # 2. 오차 계산 (Task Space Error)
        pos_err = target_pos - current_pos
        
        r_curr = R.from_quat(current_rot)
        r_targ = R.from_quat(target_rot)
        r_err = r_targ * r_curr.inv()
        rot_err_axis_angle = r_err.as_rotvec()
        
        dx = np.concatenate([pos_err, rot_err_axis_angle])
        dx_tensor = torch.tensor(dx, dtype=torch.float32).unsqueeze(-1)

        # 💡 [핵심 수정] 정수형 핸들(ID)을 기반으로 해당 액터의 등록 이름(문자열)을 추출합니다.
        actor_name = self.gym.get_actor_name(self.env, self.handle)

        # 💡 [핵심 수정] self.handle 대신 추출한 문자열(actor_name)을 세 번째 인자로 전달합니다.
        _jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, actor_name)
        jacobian = gymtorch.wrap_tensor(_jacobian_tensor)
        
        # 펌프(엔드이펙터) 링크에 해당하는 6x7 행렬 추출
        J = jacobian[0, self.pump_index - 1, :, :7].cpu()
        
        # 4. 댐핑 의사역행렬 관절 변화량 역산
        lambda_val = 0.02
        J_pinv = J.t() @ torch.inverse(J @ J.t() + lambda_val**2 * torch.eye(6))
        dq = (J_pinv @ dx_tensor).squeeze().numpy()
        
        # 5. 💡 [핵심 수정] 
        # get_actor_dof_positions 대신 정식 API인 get_actor_dof_states를 호출합니다.
        # 이 함수는 (position, velocity) 쌍으로 이루어진 구조체 배열을 반환합니다.
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
        
        # 반환된 구조체 배열에서 'position' 필드만 넘파이 배열로 추출합니다.
        current_joints = dof_states['pos']
        next_joints = current_joints + dq
        next_joints = np.clip(next_joints, self.dof_lower_limits, self.dof_upper_limits)
        
        return next_joints
    
    def move_to_ik_target(self, current_key, next_key, intra_step):
        """ 두 3D 목표 좌표 사위를 보간한 뒤 IK를 풀어 조인트 타겟을 전달하는 함수 """
        p_start = np.array(self.states[current_key][:3])
        p_end = np.array(self.states[next_key][:3])
        
        r_start = self.states[current_key][3:]
        r_end = self.states[next_key][3:]
        
        # 위치 선형 보간 (Lerp)
        interp_pos = p_start + (p_end - p_start) * intra_step
        
        # 회전 구면 선형 보간 (Slerp)
        key_rots = R.from_quat([r_start, r_end])
        slerp = Slerp([0, 1], key_rots)
        interp_rot = slerp([intra_step])[0].as_quat()
        
        # IK 연산 수행
        ik_joint_targets = self.solve_ik(interp_pos, interp_rot)
        
        # 로봇 관절 타겟 주입
        self.gym.set_actor_dof_position_targets(self.env, self.handle, ik_joint_targets)

    def move_to_joint_target(self, current_key, next_key, intra_step):
        """1. 지정한 관절 위치 사이를 선형 보간하여 이동하는 함수"""
        p_start = np.array(self.states[current_key], dtype=np.float32)
        p_end = np.array(self.states[next_key], dtype=np.float32)
        targets = p_start + (p_end - p_start) * intra_step
        
        self.gym.set_actor_dof_position_targets(self.env, self.handle, targets)

    def get_end_effector_pose(self):
        """엔드이펙터(펌프)의 1.6 스케일이 반영된 툴 끝단(Shifted) 좌표와 회전을 반환"""
        rb_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        pump_pose = rb_states['pose'][self.pump_index]
        
        pump_pos = pump_pose['p']
        pump_rot = pump_pose['r']

        # 펌프의 사원수를 회전 행렬로 변환
        quat = [pump_rot[0], pump_rot[1], pump_rot[2], pump_rot[3]]
        rot_matrix = R.from_quat(quat).as_matrix()

        # 로컬 Z축 방향 벡터 추출
        local_z_direction = rot_matrix[:, 2]

        # 💡 [1.6 스케일 핵심 보정] 로봇이 1.6배 커졌으므로, 툴 엔드이펙터 오프셋 거리도 1.6배 늘어나야 합니다.
        # 기본 거리 0.14m * 1.6 = 0.224m
        base_distance = 0.14
        scaled_distance = base_distance * self.scale
        offset = local_z_direction * scaled_distance

        p_shifted = gymapi.Vec3(
            pump_pos[0] + offset[0],
            pump_pos[1] + offset[1],
            pump_pos[2] + offset[2]
        )
        r_shifted = gymapi.Quat(pump_rot[0], pump_rot[1], pump_rot[2], pump_rot[3])
        
        return gymapi.Transform(p=p_shifted, r=r_shifted)

    def draw_debug_sphere(self, viewer, target_transform, radius=0.05):
        """2. 그랩 부위에 녹색 디버그 와이어프레임 스피어를 그리는 함수"""
        # 💡 로봇이 1.6배 커졌으므로 감지/디버그 구체 반경도 스케일에 맞춥니다.
        scaled_radius = radius * self.scale
        
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius=scaled_radius, num_lats=8, num_lons=8, color=(0, 1, 0)
        )
        gymutil.draw_lines(sphere_geom, self.gym, viewer, self.env, target_transform)

    def update_grab_logic(self, target_transform, box_handle):
        """3. 진공 흡착 그랩 및 Lerp/Slerp 스냅 상태 보간 함수"""
        # 디버거 구체 중심점 추출
        sphere_center = np.array([target_transform.p.x, target_transform.p.y, target_transform.p.z])

        # 상자의 현재 물리 상태 가져오기
        box_states = self.gym.get_actor_rigid_body_states(self.env, box_handle, gymapi.STATE_ALL)
        obj_pos = box_states['pose']['p'][0]
        obj_rot = box_states['pose']['r'][0]
        obj_center = np.array([obj_pos['x'], obj_pos['y'], obj_pos['z']])

        # 거리 계산
        distance = np.linalg.norm(sphere_center - obj_center)
        
        # 💡 스케일업에 대응하여 흡착 감지 반경도 1.6배 확장 (0.01m * 1.6 = 0.016m)
        detection_radius = 0.01 * self.scale

        # [상태 제어] 픽업 시퀀스 단계(1~3번 상태) 내에 사정거리 진입 시 흡착 기동
        if distance <= detection_radius and not self.is_attached and not self.is_snapping and self.state in range(1, 4):
            self.is_snapping = True
            self.snap_frame_count = 0
            self.snap_start_pos = np.array([obj_pos['x'], obj_pos['y'], obj_pos['z']])
            self.snap_start_rot = np.array([obj_rot['x'], obj_rot['y'], obj_rot['z'], obj_rot['w']])
            print("[그랩 컨트롤러] 물체 흡착 시퀀스 가동 (Snap Start)")

        # release(6) 상태 도달 시 진공 해제
        if self.state == 6:
            if self.is_attached or self.is_snapping:
                self.is_attached = False
                self.is_snapping = False
                print("[그랩 컨트롤러] 진공 펌프 압력 해제 (Release)")

        # 스냅 애니메이션 진행 또는 완전 흡착 유지 상태 처리
        if self.is_snapping or self.is_attached:
            target_pos = sphere_center
            target_rot = np.array([target_transform.r.x, target_transform.r.y, target_transform.r.z, target_transform.r.w])

            if self.is_snapping:
                self.snap_frame_count += 1
                t = self.snap_frame_count / self.SNAP_DURATION
                
                # 위치 선형 보간 (Lerp)
                current_snap_pos = self.snap_start_pos + (target_pos - self.snap_start_pos) * t
                
                # 회전 구면 선형 보간 (Slerp)
                rots = R.from_quat([self.snap_start_rot, target_rot])
                slerp = Slerp([0, 1], rots)
                current_snap_rot = slerp([t])[0].as_quat()

                if self.snap_frame_count >= self.SNAP_DURATION:
                    self.is_snapping = False
                    self.is_attached = True
                    print("[그랩 컨트롤러] 흡착 완료, 완벽 고정 상태 진입")
            else:
                # 완전 고정 상태
                current_snap_pos = target_pos
                current_snap_rot = target_rot

            # 계산된 좌표 물체에 덮어쓰기 및 속도 댐핑
            for i in range(len(box_states)):
                box_states['pose']['p'][i] = (current_snap_pos[0], current_snap_pos[1], current_snap_pos[2])
                box_states['pose']['r'][i] = (current_snap_rot[0], current_snap_rot[1], current_snap_rot[2], current_snap_rot[3])
                box_states['vel']['linear'][i] = (0.0, 0.0, 0.0)
                box_states['vel']['angular'][i] = (0.0, 0.0, 0.0)

            self.gym.set_actor_rigid_body_states(self.env, box_handle, box_states, gymapi.STATE_ALL)

    def update(self, frame_count, viewer, box_handle):
        """메인 루프에서 매 프레임 호출되어 전체 시퀀스를 갱신하는 최상위 함수"""
        if self.done:
            return False

        step = (frame_count // self.step_duration)
        intra_step = (frame_count % self.step_duration) / float(self.step_duration)
        is_loaded_signal = False
        
        state_keys = ["ready", "pick", "close", "lift", "rotate", "place", "release", "lift_up", "ready"]
        # state_keys = ["straight", "straight"]

        if step < len(state_keys) - 1:
            current_key = state_keys[step]
            next_key = state_keys[step + 1]
            self.state = step

            if next_key == "lift_up" and intra_step >= 0.99:
                is_loaded_signal = True
            
            # 1. 조인트 지정 이동 함수 실행
            self.move_to_joint_target(current_key, next_key, intra_step)
        else:
            self.state = 0
            self.done = True

        # 2. 엔드이펙터 포즈 연산 및 디버그 스피어 그리기
        tool_transform = self.get_end_effector_pose()
        self.draw_debug_sphere(viewer, tool_transform)

        # 3. 흡착/그랩 연산 갱신
        self.update_grab_logic(tool_transform, box_handle)

        return is_loaded_signal
    
    def update_ik(self, frame_count, viewer, box_handle):
        if self.done:
            return False

        step = (frame_count // self.step_duration)
        intra_step = (frame_count % self.step_duration) / float(self.step_duration)
        is_loaded_signal = False
        
        # IK 좌표 흐름 시퀀스 설계
        state_keys = ["straight", "move_front", "move_back", "move_left", "move_right", "straight"]
        
        if step < len(state_keys) - 1:
            current_key = state_keys[step]
            next_key = state_keys[step + 1]
            self.state = step

            if next_key == "place" and intra_step >= 0.99:
                is_loaded_signal = True
            
            # 💡 기존의 각도 제어 대신 IK 보간 제어 함수 호출
            self.move_to_ik_target(current_key, next_key, intra_step)
        else:
            self.state = 0
            self.done = True

        tool_transform = self.get_end_effector_pose()
        self.draw_debug_sphere(viewer, tool_transform)
        self.update_grab_logic(tool_transform, box_handle)

        return is_loaded_signal

    def reset(self):
        """외부 리셋 키(R) 입력 시 컨트롤러 상태를 초기화하는 편의 메서드"""
        self.done = False
        self.state = -1
        self.is_attached = False
        self.is_snapping = False
        self.snap_frame_count = 0