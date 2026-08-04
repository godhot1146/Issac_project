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

        self.base_index = gym.find_actor_rigid_body_index(env, handle, "panda_link0", gymapi.DOMAIN_ACTOR)
        if self.base_index == -1:
            print("[경고] panda_link0 링크를 찾을 수 없습니다. URDF를 확인하세요.")

        # 진공 흡착(Snap) 관련 변수 내부 탑재
        self.is_attached = False
        self.is_snapping = False
        self.snap_frame_count = 0
        self.SNAP_DURATION = 15
        
        self.snap_start_pos = None
        self.snap_start_rot = None

        self.snap_local_pos_offset = None
        self.snap_local_rot_offset = None

        self.num_dofs = 0

        self.is_holding = False 

        # 💡 [추가] 조인트(DOF) 인덱스 및 제어 모드 설정
        # URDF 상 첫 번째 회전 조인트인 'panda_joint1'의 DOF 인덱스를 찾습니다.
        self.joint1_dof_index = gym.find_actor_dof_index(env, handle, "panda_joint1", gymapi.DOMAIN_ACTOR)
        
        # 안전을 위해 joint1의 하드웨어 리밋 각도 저장 (URDF 기준 -2.8973 ~ 2.8973 라디안)
        self.joint1_lower_limit = -6.28 #-2.8973
        self.joint1_upper_limit = 6.28 #2.8973

        # 💡 시퀀스 동작 정의 (첫 코드의 7개 관절 구동 구조와 두 번째 코드의 그리퍼 포함 9개 구조를 유연하게 호환)
        # 만약 로봇이 그리퍼 2개를 포함해 총 9개 DOF를 가진 경우를 기준으로 작성되었습니다.
        self.states = {
            # "ready":   [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            "ready":   [0.0, -0.500, 0.0, -2.000, 0.0, 1.500, 0.900, 0.00, 0.00],
            "pick":    [0.0,  0.800, 0.0, -1.750, 0.0, 2.500, 0.900, 0.04, 0.04],
            "close":   [0.0,  0.800, 0.0, -1.750, 0.0, 2.500, 0.900, 0.00, 0.00],
            "lift":    [0.0, -0.500, 0.0, -2.000, 0.0, 1.500, 0.900, 0.00, 0.00],
            "rotate":  [1.571, -0.500, 0.0, -2.000, 0.0, 2.500, 0.900, 0.00, 0.00],
            "place":   [1.571,  0.350, 0.0, -1.450, 0.0, 2.500, 0.600, 0.00, 0.00],
            "release": [1.571,  0.350, 0.0, -1.450, 0.0, 2.500, 0.600, 0.04, 0.04],
            "lift_up": [1.571, -0.500, 0.0, -1.450, 0.0, 1.500, 0.600, 0.04, 0.04]
        }
        self.done = False

        self.start_step = 0  # 시작하고 싶은 단계 (기본값: 0)
        self.end_step = 7    # 끝내고 싶은 단계 (기본값: 7)
        self.pause_duration = 50

        self.ee_z_offset = 0.15

        # 💡 [내부 프레임 및 상태 관리 변수 추가]
        self.internal_frame = 0
        self.current_mode = None  # 예: "DOWN", "UP", "ROTATE" 등 모니터링용

        self._setup_dof_properties()

    def draw_z_rotational_circle(self, viewer, local_xyz, num_spheres=16, sphere_radius=0.03, color=(0.0, 0.0, 1.0)):
        """
        로봇 베이스 기준 로컬 좌표(local_xyz)를 지나며, 베이스 Z축을 중심으로 회전하는 궤적 위에 
        일정 간격으로 디버그 구체를 배치하여 원을 시각화합니다.
        
        :param viewer: Isaac Gym의 viewer 핸들
        :param local_xyz: [x, y, z] 형태의 베이스 기준 상대 좌표
        :param num_spheres: 원 궤적 위에 배치할 디버그 구체의 총 개수 (기본값: 16개)
        :param sphere_radius: 배치될 디버그 구체 각각의 반지름 (기본값: 3cm)
        :param color: 구체들의 색상 RGB (기본값: Blue)
        """
        if self.base_index == -1:
            return

        # 1. 입력받은 로컬 좌표에서 Z축 회전 반경(반지름 R)과 로컬 높이(Z) 추출
        local_x, local_y, local_z = local_xyz[0], local_xyz[1], local_xyz[2]
        radius = np.sqrt(local_x**2 + local_y**2)

        # 2. 로봇 베이스(panda_link0)의 현재 월드 포즈 가져오기
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)
        base_pose = body_states['pose'][self.base_index]
        
        base_p = np.array([base_pose['p']['x'], base_pose['p']['y'], base_pose['p']['z']])
        base_q = [base_pose['r']['x'], base_pose['r']['y'], base_pose['r']['z'], base_pose['r']['w']]
        base_rotation = R.from_quat(base_q)

        # 4. 루프 외부에서 단 한 번만 구체 지오메트리를 생성 (메모리 및 오버헤드 최적화)
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius=sphere_radius, num_lats=6, num_lons=6, color=color
        )

        # 5. 원주 위를 분할하여 일정 간격으로 구체 배치 및 렌더링
        for i in range(num_spheres):
            # 각도 계산 (360도를 num_spheres 개수로 등분)
            theta = (2.0 * np.pi * i) / num_spheres
            
            # 베이스 로컬 좌표계 기준의 구체 원점 계산
            px = radius * np.cos(theta)
            py = radius * np.sin(theta)
            pz = local_z
            local_point = np.array([px, py, pz])

            # 로컬 점을 베이스의 현재 위치/회전을 반영하여 월드 좌표로 변환
            world_point = base_rotation.apply(local_point) + base_p

            # 변환 개체(Transform) 설정
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(world_point[0], world_point[1], world_point[2])
            transform.r = gymapi.Quat(0, 0, 0, 1)

            # 현재 환경 내에 구체 드로우
            gymutil.draw_lines(sphere_geom, self.gym, viewer, self.env, transform)

    def _setup_dof_properties(self):
        """1.6배 증가한 관성을 버틸 수 있도록 강성과 감쇠, 최대 토크를 튜닝합니다."""
        dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)

        self.num_dofs = len(dof_props)
        
        # 기본 구동 모드를 위치 제어(POS)로 지정
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        
        # 💡 1.6배 덩치에 맞춰 단단하게 상향된 Stiffness와 Damping 배열 (총 7개 조인트 순서)
        # 특히 중력을 많이 받는 하부 및 중단 Y축 관절(joint2, joint4)의 값을 크게 확보합니다.
        scaled_stiffness = [4500.0, 5000.0, 4000.0, 4500.0, 2000.0, 1500.0, 800.0]
        scaled_damping   = [250.0,  300.0,  200.0,  250.0,  100.0,  80.0,   30.0]
        
        # 7개 관절에 값 매핑 및 모터 토크 한계값(effort) 강제 확장
        for i in range(min(len(dof_props), 7)):
            dof_props["stiffness"][i] = scaled_stiffness[i]
            dof_props["damping"][i] = scaled_damping[i]
            
            # 💡 로봇이 커진 만큼 모터 힘의 한계치도 최소 3배 이상 확장하여 처짐을 방지합니다.
            dof_props["effort"][i] *= 3.5 
            
        # 변경된 설정을 액터에 즉시 적용
        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)
        print("[알림] 1.6x 스케일 보정용 고강성 DOF 프로퍼티가 정상 로드되었습니다.")

        self.target_q_cached = None    # 처음에 딱 한 번만 계산해서 저장할 최종 목표 각도 배열
        self.current_y_stage = 0       # 0: Z축 회전, 1: joint2 회전, 2: joint4 회전, 3: joint6 회전, 4: 이동 완료
        self.done = False

    def get_end_effector_pose(self):
        """
        현재 엔드이펙터(펌프)의 세계 좌표계(World Frame) 기준 위치와 자세를 반환합니다.
        
        :return: (position, rotation) 형태의 튜플
                 - position: [x, y, z] 형태의 구조체 혹은 배열
                 - rotation: [x, y, z, w] 형태의 쿼터니언(Quaternion) 구조체
        """
        # 1. 안전 장치: 엔드이펙터 인덱스를 정상적으로 찾지 못한 경우 예외 처리
        if self.pump_index == -1:
            print("[에러] 엔드이펙터 링크 인덱스가 유효하지 않아 포즈를 가져올 수 없습니다.")
            return None, None

        # 2. 현재 환경(env)에 속한 전체 링크(Rigid Bodies)의 상태 배열을 가져옵니다.
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)
        
        # 3. 그 중 펌프(엔드이펙터)의 인덱스에 해당하는 포즈(pose) 데이터만 추출
        # body_states['pose']는 'p'(position)와 'r'(rotation)을 멤버로 가집니다.
        ee_pose = body_states['pose'][self.pump_index]
        
        # ee_pose.p (x, y, z), ee_pose.r (x, y, z, w 쿼터니언) 반환
        return ee_pose['p'], ee_pose['r']

    def draw_ee_debug_sphere(self, viewer, radius=0.05, color=(1.0, 0.0, 0.0)):
        """
        현재 엔드이펙터(EE) 포즈 위치에서 로컬 오프셋만큼 떨어진 지점에 디버그 구체를 그립니다.
        """
        # 1. 엔드이펙터의 현재 세계 위치(ee_pos)와 회전(ee_rot) 가져오기
        ee_pos, ee_rot = self.get_end_effector_pose()
        if ee_pos is None:
            return

        # 💡 [보정 핵심] ee_rot은 NumPy 구조체(numpy.void)이므로 딕셔너리 형태로 인덱싱해야 합니다.
        q = [ee_rot['x'], ee_rot['y'], ee_rot['z'], ee_rot['w']]
        r_matrix = R.from_quat(q)
        
        # 로컬 오프셋을 현재 EE 회전행렬과 곱해 월드 기준 변위(World Displacement)로 변환
        offset_local=[0.0, 0.0, self.ee_z_offset]
        world_offset = r_matrix.apply(offset_local)

        # 💡 ee_pos 역시 NumPy 구조체이므로 ['x'], ['y'], ['z'] 형태로 접근합니다.
        target_p = gymapi.Vec3(
            ee_pos['x'] + world_offset[0],
            ee_pos['y'] + world_offset[1],
            ee_pos['z'] + world_offset[2]
        )

        # 4. 변환 개체(Transform) 생성
        transform = gymapi.Transform()
        transform.p = target_p
        transform.r = gymapi.Quat(0, 0, 0, 1)

        # 6. 와이어프레임 지오메트리 생성 및 드로우
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius=radius, num_lats=10, num_lons=10, color=color
        )
        gymutil.draw_lines(sphere_geom, self.gym, viewer, self.env, transform)

    def get_sphere_position_relative_to_base(self):
        """
        로봇 베이스 링크(panda_link0)를 원점(0,0,0)으로 삼았을 때,
        디버그 구체 중심점의 상대적 X, Y, Z 좌표를 반환합니다.
        """
        if self.pump_index == -1 or self.base_index == -1:
            return None

        # 1. 현재 모든 링크의 포즈 가져오기
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)
        
        # 2. 베이스(link0)와 엔드이펙터(EE)의 월드 포즈 분리
        base_pose = body_states['pose'][self.base_index]
        ee_pose = body_states['pose'][self.pump_index]

        # 3. 먼저 구체의 [월드 좌표]를 계산
        ee_q = [ee_pose['r']['x'], ee_pose['r']['y'], ee_pose['r']['z'], ee_pose['r']['w']]
        ee_rotation = R.from_quat(ee_q)
        offset_local=[0.0, 0.0, self.ee_z_offset]
        world_offset = ee_rotation.apply(offset_local)
        
        sphere_world = np.array([
            ee_pose['p']['x'] + world_offset[0],
            ee_pose['p']['y'] + world_offset[1],
            ee_pose['p']['z'] + world_offset[2]
        ])

        # 4. 💡 [핵심] 베이스 포즈를 기준으로 역변환(Inverse Transform) 적용
        # 베이스의 월드 위치 및 회전행렬 구하기
        base_p = np.array([base_pose['p']['x'], base_pose['p']['y'], base_pose['p']['z']])
        base_q = [base_pose['r']['x'], base_pose['r']['y'], base_pose['r']['z'], base_pose['r']['w']]
        base_rotation = R.from_quat(base_q)

        # 상대 좌표 = (베이스 회전의 역행렬) * (구체 월드 좌표 - 베이스 월드 위치)
        relative_pos = base_rotation.inv().apply(sphere_world - base_p)

        return relative_pos  # [x, y, z] 형태의 NumPy 배열 반환

    def rotate_link1(self, target_angle_rad):
        """
        link1(panda_joint1)을 지정된 라디안 각도로 회전시키는 함수.
        💡 내부 프레임 카운트를 체크하여 반환합니다.
        """
        # 1. 회전 모드로 진입 시 프레임 자동 초기화 관리
        expected_mode = f"ROTATE_{target_angle_rad}"
        if self.current_mode != expected_mode:
            self.current_mode = expected_mode
            self.internal_frame = 0

        # 2. 목표 각도 클리핑 및 타겟 배열 가져오기
        clipped_angle = max(self.joint1_lower_limit, min(target_angle_rad, self.joint1_upper_limit))
        dof_targets = self.gym.get_actor_dof_position_targets(self.env, self.handle)
        
        if dof_targets is None or len(dof_targets) == 0:
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            dof_targets = [state[0] for state in dof_states]
        else:
            dof_targets = list(dof_targets)

        # 3. 1번 조인트 각도 수정 및 시뮬레이션 적용
        dof_targets[self.joint1_dof_index] = clipped_angle
        targets_tensor = torch.tensor(dof_targets, dtype=torch.float32, device='cpu')
        self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)
        
        # 💡 [핵심 보정] 현재 누적된 프레임 수를 임시 저장한 후 카운트를 올리고 반환합니다.
        current_run_frame = self.internal_frame
        self.internal_frame += 1
        
        return current_run_frame  # 메인 루프의 rot_frame으로 이 값이 들어갑니다.
    
    def update(self, frame_count, step_duration=100):
        if self.done:
            return False

        # 한 단계를 수행하는 데 걸리는 총 주기 (동작 시간 + 대기 시간)
        pause_dur = getattr(self, 'pause_duration', 0)
        total_cycle = step_duration + pause_dur

        # 1. 총 주기를 바탕으로 현재 주기(cycle) 내에서의 진행 상황 계산
        cycle_idx = frame_count // total_cycle
        intra_cycle_frame = frame_count % total_cycle  # 현재 주기 안에서 몇 번째 프레임인지
        
        # 2. 시작 단계 보정 반영하여 현재 step 결정
        step = cycle_idx
        if hasattr(self, 'start_step'):
            step += self.start_step

        # 3. 💡 [정지(Pause) 로직] 동작 시간(step_duration)을 넘겼고 대기 시간이 남았다면?
        # 다음 단계로 넘기지 않고, 현재 step 상태를 유지하며 강제로 멈춥니다.
        if intra_cycle_frame >= step_duration:
            # 대기 중임을 알리고 싶다면 여기에 print를 넣을 수 있습니다.
            pass 

        # 4. [끝 단계 중단점] 지정한 끝 단계에 도달하면 고정
        if hasattr(self, 'end_step'):
            if step >= self.end_step:
                step = self.end_step
                
        is_loaded_signal = False
        
        # 1단계씩 타겟 매핑
        if step == 0:    targets = self.states["ready"]
        elif step == 1:  targets = self.states["pick"]
        elif step == 2:  targets = self.states["close"]
        elif step == 3:  targets = self.states["lift"]
        elif step == 4:  targets = self.states["rotate"]
        elif step == 5:  targets = self.states["place"]
        elif step == 6:  targets = self.states["release"]
        elif step == 7:  
            targets = self.states["lift_up"]
            is_loaded_signal = True  
        else:
            targets = self.states["ready"]
            self.done = True
            print("[알림] 모든 Franka 제어 시퀀스가 완료되었습니다.")
            
        targets = targets[:self.num_dofs]
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device='cpu')
        self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)
        
        return is_loaded_signal
    
    def _send_target(self, state_key):
        """
        공통 텐서 전송 헬퍼 함수
        💡 1번 조인트(panda_joint1)의 현재 설정된 목표 각도를 유지하면서 
           나머지 관절들의 포즈(ready, pick, close, lift 등)만 덮어씁니다.
        """
        # 1. 시뮬레이션에 현재 설정되어 있는 전체 조인트의 목표(target) 각도 배열을 가져옵니다.
        current_targets = self.gym.get_actor_dof_position_targets(self.env, self.handle)
        
        # 만약 설정된 목표 각도가 없다면 현재 로봇의 실제 관절 각도를 기반으로 베이스라인을 만듭니다.
        if current_targets is None or len(current_targets) == 0:
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            current_targets = [state[0] for state in dof_states]
        else:
            current_targets = list(current_targets)

        # 2. 템플릿(ready, pick 등)에서 변경할 조인트 목표 값들을 가져옵니다.
        template_targets = self.states[state_key][:self.num_dofs]

        # 3. 💡 [핵심] 1번 조인트(joint1_dof_index)의 각도는 기존의 회전 목표를 유지하고,
        #    나머지 인덱스의 관절들만 템플릿 포즈 값으로 교체합니다.
        for i in range(len(template_targets)):
            if i == self.joint1_dof_index:
                # 1번 조인트 자리는 건드리지 않고 건너뜁니다 (기존 rotate_link1으로 먹여둔 각도 유지)
                continue
            current_targets[i] = template_targets[i]

        # 4. 최종 결합된 타겟 배열을 시뮬레이션에 주입합니다.
        targets_tensor = torch.tensor(current_targets, dtype=torch.float32, device='cpu')
        self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)

    def move_down_sequence(self, step_duration=100):
        """[내부 카운터 버전 하강 함수] 외부 인자(action_frame) 불필요"""
        # 모드가 바뀌는 순간 카운터 리셋
        if self.current_mode != "DOWN":
            self.current_mode = "DOWN"
            self.internal_frame = 0

        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle

        # 매 틱마다 내부 프레임 카운트업 진행
        self.internal_frame += 1

        if step >= 2:
            return True

        if intra_frame < step_duration:
            if step == 0:
                self._send_target("ready")
            elif step == 1:
                self._send_target("pick")
                
        return False

    def move_up_sequence(self, step_duration=100):
        """[내부 카운터 버전 상승 함수] 외부 인자(action_frame) 불필요"""
        if self.current_mode != "UP":
            self.current_mode = "UP"
            self.internal_frame = 0

        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle

        self.internal_frame += 1

        if step >= 2:
            return True

        if intra_frame < step_duration:
            if step == 0:
                self._send_target("close")
            elif step == 1:
                self._send_target("lift")
                
        return False

    def return_to_ready(self):
        self.current_mode = "READY"
        self.internal_frame = 0
        self._send_target("ready")

    def attach(self, target_actor_handle, distance_threshold=0.15):
        """
        [진공 흡착 함수] 디버그 구체와 대상 물체의 현재 '상대적 거리 및 자세'를 유지하며 흡착합니다.
        """
        if self.is_attached:
            return True

        # 1. 현재 디버그 구체의 월드 좌표 및 회전 가져오기
        ee_pos, ee_rot = self.get_end_effector_pose()
        if ee_pos is None:
            return False

        ee_q = [ee_rot['x'], ee_rot['y'], ee_rot['z'], ee_rot['w']]
        ee_rotation = R.from_quat(ee_q)
        offset_local = [0.0, 0.0, self.ee_z_offset]
        world_offset = ee_rotation.apply(offset_local)
        
        sphere_world_pos = np.array([
            ee_pos['x'] + world_offset[0],
            ee_pos['y'] + world_offset[1],
            ee_pos['z'] + world_offset[2]
        ])

        # 2. 대상 물체(박스)의 현재 월드 포즈 가져오기
        box_states = self.gym.get_actor_rigid_body_states(self.env, target_actor_handle, gymapi.STATE_POS)
        box_pos = box_states['pose'][0]['p']
        box_rot = box_states['pose'][0]['r']
        box_world_pos = np.array([box_pos['x'], box_pos['y'], box_pos['z']])
        box_world_rot = R.from_quat([box_rot['x'], box_rot['y'], box_rot['z'], box_rot['w']])

        # 3. 디버그 구체 중심과 박스 중심 간의 거리 계산
        distance = np.linalg.norm(sphere_world_pos - box_world_pos)

        # 4. 거리 조건 만족 시 상대적 오프셋 계산 후 고정
        if distance <= distance_threshold:
            self.is_attached = True
            self.attached_actor_handle = target_actor_handle
            
            # 💡 [핵심] 구체 좌표계를 기준으로 한 박스의 상대적 로컬 위치 오프셋 계산
            # 상대 위치 벡터 = 박스 월드 위치 - 구체 월드 위치
            relative_world_pos = box_world_pos - sphere_world_pos
            # 구체 회전의 역행렬을 곱해 구체 로컬 기준의 오프셋으로 변환
            self.snap_local_pos_offset = ee_rotation.inv().apply(relative_world_pos)
            
            # 💡 구체 좌표계를 기준으로 한 박스의 상대적 로컬 회전 오프셋 계산
            # 상대 회전 = (구체 회전 역행렬) * 박스 회전
            self.snap_local_rot_offset = ee_rotation.inv() * box_world_rot

            print(f"[진공 흡착 성공] 현재 거리({distance:.4f}m) 및 자세 오프셋을 유지하며 고정합니다.")
            return True
        else:
            print(f"[흡착 실패] 구체와의 거리가 너무 멉니다: {distance:.4f}m")
            return False

    def detach(self):
        """[탈착 함수] 진공을 해제하여 흡착되어 있던 물체를 떨어뜨립니다."""
        if self.is_attached:
            self.is_attached = False
            self.attached_actor_handle = None
            print("[진공 탈착] 흡착 상태를 해제했습니다.")
            return True
        return False

    def update_snapping_object(self):
        """
        [실시간 오프셋 유지 동기화 헬퍼] 
        물체가 흡착 상태일 때, 흡착 당시의 상대적 거리와 제자리 회전을 유지하며 동기화합니다.
        """
        if not self.is_attached or getattr(self, 'attached_actor_handle', None) is None:
            return

        # 1. 현재 엔드이펙터의 실시간 월드 위치와 회전 가져오기
        ee_pos, ee_rot = self.get_end_effector_pose()
        if ee_pos is None:
            return

        ee_q = [ee_rot['x'], ee_rot['y'], ee_rot['z'], ee_rot['w']]
        ee_rotation = R.from_quat(ee_q)
        
        # 2. 현재 디버그 구체의 실시간 월드 위치 계산
        offset_local = [0.0, 0.0, self.ee_z_offset]
        world_offset = ee_rotation.apply(offset_local)
        sphere_world_pos = np.array([
            ee_pos['x'] + world_offset[0],
            ee_pos['y'] + world_offset[1],
            ee_pos['z'] + world_offset[2]
        ])

        # 3. 💡 [위치 계산] 실시간 구체 위치에 흡착 당시 저장해둔 로컬 위치 오프셋을 현재 회전에 맞춰 적용
        current_box_world_pos = sphere_world_pos + ee_rotation.apply(self.snap_local_pos_offset)
        
        # 4. 💡 [회전 계산] 현재 구체 회전에 흡착 당시 저장해둔 로컬 회전 오프셋을 곱해 실시간 월드 회전 계산
        current_box_world_rot = ee_rotation * self.snap_local_rot_offset
        box_q = current_box_world_rot.as_quat() # [x, y, z, w]

        # 5. 박스 액터 상태 구조체 가져와 주입
        box_states = self.gym.get_actor_rigid_body_states(self.env, self.attached_actor_handle, gymapi.STATE_ALL)
        
        box_states['pose'][0]['p']['x'] = current_box_world_pos[0]
        box_states['pose'][0]['p']['y'] = current_box_world_pos[1]
        box_states['pose'][0]['p']['z'] = current_box_world_pos[2]
        
        box_states['pose'][0]['r']['x'] = box_q[0]
        box_states['pose'][0]['r']['y'] = box_q[1]
        box_states['pose'][0]['r']['z'] = box_q[2]
        box_states['pose'][0]['r']['w'] = box_q[3]
        
        # 속도 초기화로 관성 노이즈 제거
        box_states['vel'][0]['linear']['x'] = 0.0
        box_states['vel'][0]['linear']['y'] = 0.0
        box_states['vel'][0]['linear']['z'] = 0.0
        box_states['vel'][0]['angular']['x'] = 0.0
        box_states['vel'][0]['angular']['y'] = 0.0
        box_states['vel'][0]['angular']['z'] = 0.0

        # 6. 변경된 상태 적용
        self.gym.set_actor_rigid_body_states(
            self.env, self.attached_actor_handle, box_states, gymapi.STATE_ALL
        )