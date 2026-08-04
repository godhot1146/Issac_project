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
        # 💡 [업데이트] 추가된 2개 조인트(sub_arm_joint, cylinder_joint)를 포함한 9-DOF 시퀀스 정의
        # 마지막 두 개 원소는 각각 [sub_arm_joint(rad), cylinder_joint(m)] 타겟입니다.
        self.states = {
            "ready":   [0.0, -0.500, 0.0, -2.000, 0.0, 1.500, 0.900, 0.00, 0.00],
            "pick":    [0.0,  0.800, 0.0, -1.750, 0.0, 2.500, 0.900, 0.00, 0.00],
            "close":   [0.0,  0.800, 0.0, -1.750, 0.0, 2.500, 0.900, 0.50, 0.02], # 예시 구동 상태 값 부가
            "lift":    [0.0, -0.500, 0.0, -2.000, 0.0, 1.500, 0.900, 0.50, 0.02],
            "rotate":  [1.571, -0.500, 0.0, -2.000, 0.0, 2.500, 0.900, 0.50, 0.02],
            "place":   [1.571,  0.350, 0.0, -1.450, 0.0, 2.500, 0.600, 0.50, 0.02],
            "release": [1.571,  0.350, 0.0, -1.450, 0.0, 2.500, 0.600, 0.00, 0.00],
            "lift_up": [1.571, -0.500, 0.0, -1.450, 0.0, 1.500, 0.600, 0.00, 0.00]
        }
        self.box_states = {
            "ready":   [0.0, 0.0, 0.0, 0.07, 0.0, 0.0, -0.6, 0.0, 0.0],
            "reach":    [0.0,  0.261, 0.0, -0.681, 0.0, 2.530, -0.78, -0.087, -1.571],
            "reach2":    [-1.571,  0.261, 0.0, -0.681, 0.0, 2.530, -0.78, -0.087, -1.571],
            "rotate1":  [-1.571,  0.261, 0.0, -0.681, 0.0, 2.530, 0.78, -0.087, -1.571],
            "rotate1_2":  [0.0,  0.261, 0.0, -0.681, 0.0, 2.530, 0.78, -0.087, -1.571],
            "foldside": [0.0,  0.261, 0.0, -0.681, 0.0, 2.530, 0.78, -0.087, 0],

            "place1_1": [-0.041635, -0.382075, 0.034077, -1.942754, 3.138309, 3.138278, -2.098, -0.087, 0],
            "place1_2": [-0.579533, -0.292520, 0.098965, -2.011047, 1.904253, 2.637398, -1.137, -0.087, 0],
            "place1_3": [-0.849335, 0.198405, -0.039074, -1.480517, 1.684732, 2.256319, -0.963, -0.087, 0],
            "place1_4": [-0.849335, 0.198405, -0.039074, -1.480517, 1.684732, 2.256319, -1.312, -0.087, 0],
            
            "place2_1": [-0.788810, 0.246529, -0.117861, -1.729725, 1.915238, 2.168021, -1.637-0.1, -0.087, 0],
            "place2_2": [-0.780077, 0.274376, -0.131552, -1.763470, 1.956165, 2.140461, -1.375-0.1, -0.087, 0],
            "place2_3": [-0.785187, 0.352556, -0.134681, -1.813778, 2.030485, 2.078846, -1.5-0.1, -0.087, 0],
            "place2_4": [-0.796159, 0.373772, -0.124168, -1.820817, 2.043594, 2.066437, -1.5-0.1, -0.087, 0],
            "place2_5": [-0.697589, 0.271896, -0.151849, -1.999657, 2.132045, 2.079133, -1.620-0.1, -0.087, 0],

            "move_1": [-0.466205, 0.032392, -0.174637, -2.297472, 2.337048, 2.182551, -1.812-0.1, -0.087, 0],
            "move_2": [-0.269394, -0.108133, -0.174808, -2.444680, 2.556385, 2.277166, -1.969-0.1, -0.087, 0],
            "move_3": [-0.084838, -0.197000, -0.144089, -2.526315, 2.834674, 2.349289, -2.150-0.1, -0.087, 0],
            "move_4": [0.091488, -0.226456, -0.090777, -2.554078, 3.135942, 2.365676, -2.367-0.1, -0.087, 0],
            
            "move_5": [-0.170951, -0.228706, 0.170340, -2.555639, 3.138772, 2.360299, -2.32-0.1, -0.087, 0],
            "move_6": [0.057997, -0.192861, 0.175225, -2.524400, 3.467046, 2.335905, -2.547-0.1, -0.087, 0],
            "move_7": [0.258201, -0.093569, 0.199891, -2.431011, 3.752986, 2.257367, -2.739-0.1, -0.087, 0],
            "move_8": [0.418430, 0.046864, 0.227105, -2.280882, 3.956348, 2.165194, -2.886-0.1, -0.087, 0],
            "move_9": [0.565184, 0.213989, 0.236687, -2.080811, 4.098564, 2.0783385, -3.018-0.1, -0.087, 0],
            "move_10": [0.630398, 0.427210, 0.316248, -1.805176, 4.180365, 1.990649, -3.190-0.1, -0.087, 0],
            "move_11": [0.763891, 0.688000, 0.320963, -1.422671, 4.236083, 1.940856, -3.357-0.1, -0.087, 0],
        }
        # move5부터 link1의 각도는 비슷하게, link5의 각도는 robodk의 각도를 변환 한 것에 2pi를 더하기
        #[그대로, 그대로, 그대로, 반전, 그대로, 반전, 직접 지정]

        self.pick_and_place_states = {
            "ready":   [0.0, 0.0, 0.0, 0.07, 0.0, 0.0, -0.6, 0.0, 0.0],
            "reach1":   [2.711, -0.2104, 1.0124, -1.6136, 0.1748, 1.4974, 0.0, 0.0, 0.0], # 2.6411, -0.5034, 1.3231, 2.2466, 0.5526, -2.0431, 0.5153
            "reach2":   [3.1413, 0.199, 0.9485, -0.7893, -0.168, 1.6406, -0.8297, 0.0, 0.0, 0.0],
            "move1":   [-0.1291, -0.1663, 0.1074, -1.2391, -0.0084, 1.0793, 0.0246, -0.6, 0.0, 0.0],
            "move2":   [-0.1617, -0.4304, 0.1153, -1.9961, 0.0226, 1.5751, 0.0536, -0.6, 0.0, 0.0],
        }
        #link4, link6, link7 반전
        self.done = False
        self.start_step = 0  # 시작하고 싶은 단계 (기본값: 0)
        self.end_step = 7    # 끝내고 싶은 단계 (기본값: 7)
        self.pause_duration = 50
        self.ee_z_offset = 0.1
        # 💡 [내부 프레임 및 상태 관리 변수 추가]
        self.internal_frame = 0
        self.current_mode = None  # 예: "DOWN", "UP", "ROTATE" 등 모니터링용
        # ==============================================================================
        # 🛠️ [추가] 컨트롤러 내부 수동 제어 전용 상태 변수 초기화
        # ==============================================================================
        self.manual_mode = False          # 수동 모드 활성화 플래그
        self.selected_dof_idx = 0         # 현재 조작 대상으로 선택된 DOF 인덱스
        self.manual_targets = np.zeros(self.num_dofs, dtype=np.float32) # 수동 타겟 배열
        self.dof_names = gym.get_actor_dof_names(env, handle)           # DOF 이름 리스트
        self.use_manual = False
        # ==============================================================================
        self.num_dofs = 9  # 혹은 자동으로 계산된 조인트 수 (len(dof_props))
        self.current_targets = np.zeros(self.num_dofs, dtype=np.float32)
        self.first_frame_lock = True  # 처음 시작 시 현재 상태를 캐싱하기 위한 플래그
        self.current_speed_gain = 1.2

        self.traj_start_q = None
        self.traj_target_q = None
        self.traj_time = 0.0
        self.traj_duration = 2.0

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
        """💡 [업데이트] 총 9개 조인트의 가변 길이를 안전하게 동적 처리하고, 추가 조인트의 강성을 셋업합니다."""
        dof_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        self.num_dofs = len(dof_props)
        
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        
        # 기본 7개 관절용 스케일 보정값
        scaled_stiffness = [4500.0, 5000.0, 4000.0, 4500.0, 2000.0, 1500.0, 800.0]
        scaled_damping   = [250.0,  300.0,  200.0,  250.0,  100.0,  80.0,   30.0]
        
        # 💡 추가 조인트 전용 파라미터 지정 (8번째: sub_arm_joint, 9번째: cylinder_joint)
        # 펌프 파츠는 본체 대비 가벼우므로 적절한 강성과 높은 댐핑을 주어 오실레이션을 방지합니다.
        extra_stiffness = [600.0, 500.0]
        extra_damping   = [40.0, 30.0]
        
        # 전체 가용 DOF 루프 처리
        for i in range(self.num_dofs):
            if i < 7:
                dof_props["stiffness"][i] = scaled_stiffness[i]
                dof_props["damping"][i] = scaled_damping[i]
                dof_props["effort"][i] *= 3.5  # 스케일업 보정
            else:
                # 💡 7번 인덱스 이후 추가된 조인트 프로퍼티 예외 처리 적용
                extra_idx = i - 7
                if extra_idx < len(extra_stiffness):
                    dof_props["stiffness"][i] = extra_stiffness[extra_idx]
                    dof_props["damping"][i] = extra_damping[extra_idx]
                    # 추가 관절의 기본 effort 한계선 유지 혹은 가중
        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)
        print(f"[알림] {self.num_dofs}-DOF 스케일 보정용 고강성 DOF 프로퍼티가정상 로드되었습니다.")
        self.target_q_cached = None    
        self.current_y_stage = 0       
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
    
    def rotate_link1(self, target_angle_rad, step_duration=100):
        """
        link1(panda_joint1)을 지정된 라디안 각도로 회전시키는 함수.
        시퀀스가 완료되면 True, 진행 중이면 False를 반환합니다.
        """
        if self.manual_mode: 
            return False
        # 1. 회전 모드로 진입 시 프레임 자동 초기화 관리
        expected_mode = f"ROTATE_{target_angle_rad}"
        if self.current_mode != expected_mode:
            self.current_mode = expected_mode
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        
        # 2. 목표 각도 클리핑 및 타겟 배열 가져오기
        clipped_angle = max(self.joint1_lower_limit, min(target_angle_rad, self.joint1_upper_limit))
        
        # 내부 보간 시스템용 current_targets가 정의되어 있다면 그것을 수정
        if hasattr(self, 'current_targets'):
            self.current_targets[self.joint1_dof_index] = clipped_angle
        else:
            # 기존 직접 주입 방식 백업 (보간을 안 쓸 경우)
            dof_targets = self.gym.get_actor_dof_position_targets(self.env, self.handle)
            if dof_targets is None or len(dof_targets) == 0:
                dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
                dof_targets = [state[0] for state in dof_states]
            else:
                dof_targets = list(dof_targets)
            dof_targets[self.joint1_dof_index] = clipped_angle
            targets_tensor = torch.tensor(dof_targets, dtype=torch.float32, device='cpu')
            self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)
        # 3. 시간(프레임) 경과 체크 및 상태 반환
        self.internal_frame += 1
        
        if self.internal_frame >= total_cycle:
            return True  # 지정된 회전 + 대기 시간이 모두 끝나면 완료 신호 리턴
            
        return False
    
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
        """💡 [업데이트] 추가된 구동 조인트 인덱스(7, 8번)들까지 누락 없이 동시 제어하도록 업데이트되었습니다."""
        if self.manual_mode: return
        current_targets = self.gym.get_actor_dof_position_targets(self.env, self.handle)
        
        if current_targets is None or len(current_targets) == 0:
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            current_targets = [state[0] for state in dof_states]
        else:
            current_targets = list(current_targets)
        template_targets = self.states[state_key][:self.num_dofs]
        # 💡 1번 조인트만 스킵하고, 새로 추가된 sub_arm_joint 및 cylinder_joint를 포함하여 타겟 매핑
        for i in range(min(len(template_targets), len(current_targets))):
            if i == self.joint1_dof_index:
                continue
            current_targets[i] = template_targets[i]
        targets_tensor = torch.tensor(current_targets[:self.num_dofs], dtype=torch.float32, device='cpu')
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
        속도(Velocity) 기반 제어를 위해 필요한 초기 오프셋을 기록합니다.
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
            
            # 구체 좌표계를 기준으로 한 박스의 상대적 로컬 위치 오프셋 계산
            relative_world_pos = box_world_pos - sphere_world_pos
            self.snap_local_pos_offset = ee_rotation.inv().apply(relative_world_pos)
            
            # 구체 좌표계를 기준으로 한 박스의 상대적 로컬 회전 오프셋 계산
            self.snap_local_rot_offset = ee_rotation.inv() * box_world_rot
            print(f"[진공 흡착 성공] 속도 제어 모드로 전환합니다. 초기 거리: {distance:.4f}m")
            return True
        else:
            print(f"[흡착 실패] 구체와의 거리가 너무 멩니다: {distance:.4f}m")
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
        Articulation 경고를 방지하기 위해 오직 Root 링크(인덱스 0)의 속도만 제어합니다.
        """
        if not self.is_attached or getattr(self, 'attached_actor_handle', None) is None:
            return
        # 1. 현재 엔드이펙터 및 디버그 구체의 실시간 월드 포즈 계산
        ee_pos, ee_rot = self.get_end_effector_pose()
        if ee_pos is None:
            return
        ee_q = [ee_rot['x'], ee_rot['y'], ee_rot['z'], ee_rot['w']]
        ee_rotation = R.from_quat(ee_q)
        
        offset_local = [0.0, 0.0, self.ee_z_offset]
        world_offset = ee_rotation.apply(offset_local)
        sphere_world_pos = np.array([
            ee_pos['x'] + world_offset[0],
            ee_pos['y'] + world_offset[1],
            ee_pos['z'] + world_offset[2]
        ])
        # 2. [목표 포즈 계산] 흡착 당시 유지하기로 했던 로컬 오프셋 반영
        target_box_pos = sphere_world_pos + ee_rotation.apply(self.snap_local_pos_offset)
        target_box_rot = ee_rotation * self.snap_local_rot_offset
        # 3. [현재 물체 포즈 가져오기]
        box_states = self.gym.get_actor_rigid_body_states(self.env, self.attached_actor_handle, gymapi.STATE_POS)
        current_box_pos = np.array([box_states['pose'][0]['p']['x'], box_states['pose'][0]['p']['y'], box_states['pose'][0]['p']['z']])
        current_box_rot = R.from_quat([box_states['pose'][0]['r']['x'], box_states['pose'][0]['r']['y'], box_states['pose'][0]['r']['z'], box_states['pose'][0]['r']['w']])
        # 4. 물리 시뮬레이션의 타임스텝(dt) 설정
        dt = self.gym.get_sim_params(self.sim).dt
        if dt <= 0: 
            dt = 1.0 / 60.0
        # 5. 🚀 [위치 및 회전 보간(Interpolation) 연산]
        # 즉시 텔레포트하면 물리 충돌 오차가 심해지므로, 현재 위치에서 목표 위치로 
        # 게인(Gain)을 이용해 한 프레임 단위로 부드럽게 추종하도록 변위를 구합니다.
        p_gain_pos = 0.5  # 다음 위치로 이동할 추종 속도 비율 (0.1 ~ 1.0 사이 조정 가능)
        next_box_pos = current_box_pos + (target_box_pos - current_box_pos) * p_gain_pos
        # ① 현재 회전과 목표 회전 사이의 [회전 오차(가야 할 회전량)]를 구합니다.
        # 수학적으로 R_error = R_target * R_current_inverse 입니다.
        rot_error = target_box_rot * current_box_rot.inv()
        # ② ✨ [핵심] 회전 오차를 '3차원 회전 벡터(Rotation Vector)'로 변환합니다.
        # 이 함수가 반환하는 벡터의 방향은 [회전 축], 크기(Magnitude)는 [회전 각도(라디안)]를 나타냅니다.
        rot_error_vec = rot_error.as_rotvec()
        # 회전 보간 (Slerp를 사용해 부드럽게 목표 쿼터니언으로 유도)
        p_gain_rot = 0.4
        key_times = [0, 1]
        key_rots = R.from_quat([current_box_rot.as_quat(), target_box_rot.as_quat()])
        slerp = Slerp(key_times, key_rots)
        next_box_rot = slerp([p_gain_rot])[0]
        box_q = next_box_rot.as_quat()
        # 6. 🔥 [핵심 제어] Transform 직접 주입 + 선속도/각속도 물리 동기화 
        box_root_env_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.attached_actor_handle, 0, gymapi.DOMAIN_ENV
        )
        # Isaac Gym Transform 구조체 생성 및 할당
        next_pose = gymapi.Transform()
        next_pose.p = gymapi.Vec3(next_box_pos[0], next_box_pos[1], next_box_pos[2])
        next_pose.r = gymapi.Quat(box_q[0], box_q[1], box_q[2], box_q[3])
        
        # 💡 1단계: 루트의 다음 프레임 위치/회전 강제 주입
        self.gym.set_rigid_transform(self.env, box_root_env_idx, next_pose)
        # 💡 2단계: 선속도 역산 (V = dX / dt)
        vel_x = (target_box_pos[0] - current_box_pos[0]) / dt
        vel_y = (target_box_pos[1] - current_box_pos[1]) / dt
        vel_z = (target_box_pos[2] - current_box_pos[2]) / dt
        self.gym.set_rigid_linear_velocity(self.env, box_root_env_idx, gymapi.Vec3(vel_x, vel_y, vel_z))
        # 💡 3단계: 각속도 역산 (Omega = dTheta / dt)
        # 5번 스텝에서 구한 rot_error_vec(라디안 단위의 회전축 벡터)을 시간(dt)으로 나누어 각속도를 구합니다.
        omega_x = rot_error_vec[0] / dt
        omega_y = rot_error_vec[1] / dt
        omega_z = rot_error_vec[2] / dt
        self.gym.set_rigid_angular_velocity(self.env, box_root_env_idx, gymapi.Vec3(omega_x, omega_y, omega_z))
    
    # ==============================================================================
    # 🛠️ [신규 추가] 외부 키보드 이벤트 연동 핸들러 및 수동 주입 통합 메서드
    # ==============================================================================
    def handle_keyboard_event(self, action):
        """메인 루프의 키보드 이벤트 분기 처리 시 호출될 함수"""
        if not self.use_manual:
            return
        # 1. 수동 조작 모드 온/오프 (M 키)
        if action == "toggle_manual_mode":
            self.manual_mode = not self.manual_mode
            mode_str = "🔥 수동(MANUAL)" if self.manual_mode else "🤖 자동(AUTOMATIC)"
            print(f"\n[Franka 제어 모드 변경] 현재 구동 모드: {mode_str}")
            
            # 수동 조작 진입 시, 로봇이 튀는 것을 방지하기 위해 현재의 조인트 각도들을 타겟 버퍼에 복사
            if self.manual_mode:
                dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
                self.manual_targets = np.array([state[0] for state in dof_states], dtype=np.float32)
                print(f" -> 현재 조작 관절 [{self.selected_dof_idx}]: {self.dof_names[self.selected_dof_idx]}")
        # 수동 모드일 때만 하위 키 바인딩 유효화
        if self.manual_mode:
            # 2. 관절(DOF) 순회 선택 (1번 / 2번 키)
            if action == "select_prev_joint":
                self.selected_dof_idx = (self.selected_dof_idx - 1) % self.num_dofs
                print(f"🔄 [관절 선택] ➡️ [{self.selected_dof_idx}]: {self.dof_names[self.selected_dof_idx]} (현재 타겟: {self.manual_targets[self.selected_dof_idx]:.3f} rad)")
            elif action == "select_next_joint":
                self.selected_dof_idx = (self.selected_dof_idx + 1) % self.num_dofs
                print(f"🔄 [관절 선택] ➡️ [{self.selected_dof_idx}]: {self.dof_names[self.selected_dof_idx]} (현재 타겟: {self.manual_targets[self.selected_dof_idx]:.3f} rad)")
            # 3. 관절 조절 제어 (3번 / 4번 키)
            elif action == "decrease_joint_angle":
                self.manual_targets[self.selected_dof_idx] -= np.radians(1.0)  # 1도 감소
                # print(f"📉 [{self.dof_names[self.selected_dof_idx]}] 타겟 변경: {self.manual_targets[self.selected_dof_idx]:.3f} rad")
            elif action == "increase_joint_angle":
                self.manual_targets[self.selected_dof_idx] += np.radians(1.0)  # 1도 증가
                # print(f"📈 [{self.dof_names[self.selected_dof_idx]}] 타겟 변경: {self.manual_targets[self.selected_dof_idx]:.3f} rad")
        # 4. 스페이스바 키 입력 시 정보 리포트 분기 처리
        if action == "start_process" and self.manual_mode:
            print(f"\n📊 [Franka 수동 조작 관절 리포트]")
            print(f" - 선택된 관절 명칭: {self.dof_names[self.selected_dof_idx]} (Index: {self.selected_dof_idx})")
            print(f" - 목적 각도: {self.manual_targets[self.selected_dof_idx]:.4f} rad ({np.degrees(self.manual_targets[self.selected_dof_idx]):.1f}°)")
    
    def apply_manual_targets(self):
        """수동 모드 활성화 시 매 프레임마다 Tensor API를 통해 목표 각도를 물리 인스턴스에 강제 주입"""
        if self.manual_mode:
            targets_tensor = torch.tensor(self.manual_targets, dtype=torch.float32, device='cpu')
            self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)
    
    def box_catch_sequence(self, step_duration=100):
        """
        [신규 추가] 상자를 잡기 위해 box_states["ready"] -> box_states["reach"] 단계를 
        순차적으로 구동하는 내부 카운터 버전 시퀀스 함수입니다.
        """
        # 수동 모드일 때는 자동 시퀀스 연산을 무시합니다.
        if self.manual_mode: 
            return False
        # 모드가 'BOX_CATCH'로 새로 진입하거나 바뀌는 순간 내부 카운터 리셋
        if self.current_mode != "BOX_CATCH":
            self.current_mode = "BOX_CATCH"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        # 매 틱마다 내부 프레임 카운트업 진행
        self.internal_frame += 1
        # 정의된 2개의 스텝(0: ready, 1: reach)을 모두 끝내면 시퀀스 완료 플래그(True) 반환
        if step >= 3:
            return True
        # 대기 시간(pause_duration)을 제외한 실제 구동 타임 프레임 내에서만 타겟 세팅
        if intra_frame < step_duration:
            if step == 0:
                self._send_box_target("ready")
            elif step == 1:
                self._send_box_target("reach")
            elif step == 2:
                self._send_box_target("reach2")
                
        return False
    
    def box_rotate_sequence(self, step_duration=100):
        """
        [시퀀스 2] 상자를 파지한 상태에서 rotate1 포즈로 회전하는 함수
        시퀀스 순서: 0: reach ➡️ 1: rotate1
        """
        if self.manual_mode: return False
        if self.current_mode != "BOX_ROTATE":
            self.current_mode = "BOX_ROTATE"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        self.internal_frame += 1
        if step >= 2:
            return True
        if intra_frame < step_duration:
            if step == 0: self._send_box_target("reach2")
            elif step == 1: self._send_box_target("rotate1")
                
        return False
    
    def box_foldside_sequence(self, step_duration=100):
        if self.manual_mode: return False, False
        if self.current_mode != "BOX_FOLDSIDE":
            self.current_mode = "BOX_FOLDSIDE"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        self.internal_frame += 1
        # 시퀀스 완전 종료 조건
        if step >= 2:
            return True, False
        should_fold_box = False
        if intra_frame < step_duration:
            if step == 0: 
                self._send_box_target("rotate1_2")
            elif step == 1: 
                self._send_box_target("foldside")
                # 🎯 step == 1이 되는 순간이 sub arm이 foldside로 움직이기 시작하는 타이밍입니다.
                # 혹은 움직임이 시작되고 약간의 딜레이를 주고 싶다면 (예: intra_frame == 10) 등으로 조절 가능합니다.
                if intra_frame == 0: 
                    should_fold_box = True
                    
        return False, should_fold_box
    
    def box_place1_sequence(self, step_duration=100):
        if self.manual_mode: return False
        if self.current_mode != "BOX_PLACE1":
            self.current_mode = "BOX_PLACE1"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        self.internal_frame += 1
        if step >= 5:
            return True
        if intra_frame < step_duration:
            if step == 0: self._send_box_target("foldside")
            elif step == 1: self._send_box_target("place1_1")
            elif step == 2: self._send_box_target("place1_2")
            elif step == 3: self._send_box_target("place1_3")
            elif step == 4: self._send_box_target("place1_4")
                
        return False
    
    def box_place2_sequence(self, step_duration=100):
        if self.manual_mode: 
            return False, None
            
        if self.current_mode != "BOX_PLACE2":
            self.current_mode = "BOX_PLACE2"
            self.internal_frame = 0
            
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        self.internal_frame += 1
        
        # 시퀀스 완전 종료 조건
        if step >= 6:
            return True, None

        # 🎯 해당 스텝이 시작되는 첫 번째 프레임(intra_frame == 0)일 때 스텝 번호를 저장
        started_step = None
        if intra_frame == 0:
            started_step = step

        if intra_frame < step_duration:
            if step == 0: self._send_box_target("place1_4")
            elif step == 1: self._send_box_target("place2_1")
            elif step == 2: self._send_box_target("place2_2")
            elif step == 3: self._send_box_target("place2_3")
            elif step == 4: self._send_box_target("place2_4")
            elif step == 5: self._send_box_target("place2_5")
                
        # (시퀀스 완료 여부, 새로 시작된 스텝 번호) 리턴
        return False, started_step
    
    def box_move_sequence(self, step_duration=100):
        if self.manual_mode: 
            return False, None
            
        if self.current_mode != "BOX_MOVE":
            self.current_mode = "BOX_MOVE"
            self.internal_frame = 0
            
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        self.internal_frame += 1
        
        if step >= 12:
            return True, None

        started_step = None
        if intra_frame == 0:
            started_step = step

        if intra_frame < step_duration:
            if step == 0: self._send_box_target("place2_5")
            elif step == 1: self._send_box_target("move_1")
            elif step == 2: self._send_box_target("move_2")
            elif step == 3: self._send_box_target("move_3")
            elif step == 4: self._send_box_target("move_4")
            elif step == 5: self._send_box_target("move_5")
            elif step == 6: self._send_box_target("move_6")
            elif step == 7: self._send_box_target("move_7")
            elif step == 8: self._send_box_target("move_8")
            elif step == 9: self._send_box_target("move_9")
            elif step == 10: self._send_box_target("move_10")
            elif step == 11: self._send_box_target("move_11")
                
        return False, started_step
    
    def catch_sequence(self, step_duration=100):
        """
        [신규 추가] 상자를 잡기 위해 box_states["ready"] -> box_states["reach"] 단계를 
        순차적으로 구동하는 내부 카운터 버전 시퀀스 함수입니다.
        """
        # 수동 모드일 때는 자동 시퀀스 연산을 무시합니다.
        if self.manual_mode: 
            return False
        # 모드가 'CATCH'로 새로 진입하거나 바뀌는 순간 내부 카운터 리셋
        if self.current_mode != "CATCH":
            self.current_mode = "CATCH"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        # 매 틱마다 내부 프레임 카운트업 진행
        self.internal_frame += 1
        # 정의된 2개의 스텝(0: ready, 1: reach)을 모두 끝내면 시퀀스 완료 플래그(True) 반환
        if step >= 4:
            return True
        # 대기 시간(pause_duration)을 제외한 실제 구동 타임 프레임 내에서만 타겟 세팅
        if intra_frame < step_duration:
            if step == 0: self._send_pick_and_place_target("ready")
            elif step == 1: self._send_pick_and_place_target("reach1")
                
        return False
    
    def catch_sequence2(self, step_duration=100):
        """
        [신규 추가] 상자를 잡기 위해 box_states["ready"] -> box_states["reach"] 단계를 
        순차적으로 구동하는 내부 카운터 버전 시퀀스 함수입니다.
        """
        # 수동 모드일 때는 자동 시퀀스 연산을 무시합니다.
        if self.manual_mode: 
            return False
        # 모드가 'CATCH'로 새로 진입하거나 바뀌는 순간 내부 카운터 리셋
        if self.current_mode != "CATCH2":
            self.current_mode = "CATCH2"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        # 매 틱마다 내부 프레임 카운트업 진행
        self.internal_frame += 1
        # 정의된 2개의 스텝(0: ready, 1: reach)을 모두 끝내면 시퀀스 완료 플래그(True) 반환
        if step >= 4:
            return True
        # 대기 시간(pause_duration)을 제외한 실제 구동 타임 프레임 내에서만 타겟 세팅
        if intra_frame < step_duration:
            if step == 0: self._send_pick_and_place_target("reach1")
            elif step == 1: self._send_pick_and_place_target("reach2")
                
        return False
    
    def catch_move_sequence(self, step_duration=100):
        """
        [신규 추가] 상자를 잡기 위해 box_states["ready"] -> box_states["reach"] 단계를 
        순차적으로 구동하는 내부 카운터 버전 시퀀스 함수입니다.
        """
        # 수동 모드일 때는 자동 시퀀스 연산을 무시합니다.
        if self.manual_mode: 
            return False
        # 모드가 'CATCH'로 새로 진입하거나 바뀌는 순간 내부 카운터 리셋
        if self.current_mode != "CATCH_MOVE":
            self.current_mode = "CATCH_MOVE"
            self.internal_frame = 0
        total_cycle = step_duration + self.pause_duration
        step = self.internal_frame // total_cycle
        intra_frame = self.internal_frame % total_cycle
        # 매 틱마다 내부 프레임 카운트업 진행
        self.internal_frame += 1
        # 정의된 2개의 스텝(0: ready, 1: reach)을 모두 끝내면 시퀀스 완료 플래그(True) 반환
        if step >= 4:
            return True
        # 대기 시간(pause_duration)을 제외한 실제 구동 타임 프레임 내에서만 타겟 세팅
        if intra_frame < step_duration:
            if step == 0: self._send_pick_and_place_target("reach2")
            elif step == 1: self._send_pick_and_place_target("move1")
            elif step == 2: self._send_pick_and_place_target("move2")
                
        return False
    
    def _send_pick_and_place_target(self, state_key):
        """box_states 전용 타겟 업데이트 헬퍼 (직접 주입하지 않고 목표치 버퍼만 갱신)"""
        # if self.manual_mode: 
        #     return
        # # 초기 1회만 현재 물리 상태를 current_targets에 동기화하여 로봇이 튀는 현상 방지
        # if self.first_frame_lock:
        #     dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
        #     self.current_targets = np.array([state[0] for state in dof_states], dtype=np.float32)
        #     self.first_frame_lock = False
        # template_targets = self.box_states[state_key][:self.num_dofs]
        # # 💡 [수정] 모든 조인트(link1 포함)의 목표치를 정상 매핑합니다.
        # for i in range(min(len(template_targets), len(self.current_targets))):
        #     self.current_targets[i] = template_targets[i]

        if self.manual_mode: 
            return
            
        # 첫 프레임 록 해제 및 초기화
        if self.first_frame_lock:
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            self.current_targets = np.array([state[0] for state in dof_states], dtype=np.float32)
            self.first_frame_lock = False

        new_target = np.array(self.pick_and_place_states[state_key][:self.num_dofs], dtype=np.float32)
        
        # 이전 목표와 새로 들어온 목표가 다를 때만 궤적 재생성
        if self.traj_target_q is None or not np.allclose(self.traj_target_q, new_target):
            # 현재의 실제 물리 상태 혹은 직전 타겟을 시작점으로 잡음
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            self.traj_start_q = np.array([state[0] for state in dof_states], dtype=np.float32)
            self.traj_target_q = new_target
            self.traj_time = 0.0  # 타이머 초기화

    
    def _send_box_target(self, state_key):
        """box_states 전용 타겟 업데이트 헬퍼 (직접 주입하지 않고 목표치 버퍼만 갱신)"""
        # if self.manual_mode: 
        #     return
        # # 초기 1회만 현재 물리 상태를 current_targets에 동기화하여 로봇이 튀는 현상 방지
        # if self.first_frame_lock:
        #     dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
        #     self.current_targets = np.array([state[0] for state in dof_states], dtype=np.float32)
        #     self.first_frame_lock = False
        # template_targets = self.box_states[state_key][:self.num_dofs]
        # # 💡 [수정] 모든 조인트(link1 포함)의 목표치를 정상 매핑합니다.
        # for i in range(min(len(template_targets), len(self.current_targets))):
        #     self.current_targets[i] = template_targets[i]

        if self.manual_mode: 
            return
            
        # 첫 프레임 록 해제 및 초기화
        if self.first_frame_lock:
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            self.current_targets = np.array([state[0] for state in dof_states], dtype=np.float32)
            self.first_frame_lock = False

        new_target = np.array(self.box_states[state_key][:self.num_dofs], dtype=np.float32)
        
        # 이전 목표와 새로 들어온 목표가 다를 때만 궤적 재생성
        if self.traj_target_q is None or not np.allclose(self.traj_target_q, new_target):
            # 현재의 실제 물리 상태 혹은 직전 타겟을 시작점으로 잡음
            dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
            self.traj_start_q = np.array([state[0] for state in dof_states], dtype=np.float32)
            self.traj_target_q = new_target
            self.traj_time = 0.0  # 타이머 초기화

    def update_joints_interpolation(self, dt=1.0/30.0):
        """
        📦 CardboardBoxManager와 동일한 속도로 프랑카 조인트를 보간 구동합니다.
        메인 루프에서 매 틱마다 시뮬레이션 전(step_graphics/simulate 전)에 호출되어야 합니다.
        """
        # if self.manual_mode or self.num_dofs == 0:
        #     return
        # # 1. 박스 매니저와 완전히 동일한 스텝 제한 필터 적용 (1.2 * dt)
        # max_step = self.current_speed_gain * dt
        
        # # 현재 실제 물리 엔진이 바라보고 있는 타겟 값을 가져오거나 상태를 기반으로 추종 변위 계산
        # # 여기서는 current_targets가 필터링된 최종 명령어가 됩니다.
        # dof_targets = self.gym.get_actor_dof_position_targets(self.env, self.handle)
        # if dof_targets is None or len(dof_targets) == 0:
        #     dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
        #     actual_targets = np.array([state[0] for state in dof_states], dtype=np.float32)
        # else:
        #     actual_targets = np.array(list(dof_targets), dtype=np.float32)
        # # 차이만큼 clip하여 부드럽게 전진
        # diff = self.current_targets - actual_targets
        # next_targets = actual_targets + np.clip(diff, -max_step, max_step)
        # # 2. 물리 인스턴스에 최종 연산된 타겟 주입
        # targets_tensor = torch.tensor(next_targets, dtype=torch.float32, device='cpu')
        # self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)

        """
        [업데이트] 5차 다항식(Quintic Spline)을 이용해 
        속도/가속도가 부드럽게 이어지는 연속 제어 신호를 물리 엔진에 주입합니다.
        """
        if self.manual_mode or self.num_dofs == 0:
            return
            
        if self.traj_start_q is not None and self.traj_target_q is not None:
            # 시간 누적
            self.traj_time += dt
            
            # 5차 다항식 보간 각도 계산
            self.current_targets = self._compute_quintic_spline(
                self.traj_start_q, 
                self.traj_target_q, 
                self.traj_time, 
                self.traj_duration
            )
            
        # 물리 인스턴스에 최종 연산된 타겟 주입
        targets_tensor = torch.tensor(self.current_targets, dtype=torch.float32, device='cpu')
        self.gym.set_actor_dof_position_targets(self.env, self.handle, targets_tensor)
    
    def compute_ik_7dof(self, target_pos, target_quat, max_iters=15, tol=1e-3, dt=0.05):
        """
        앞의 7개 관절만을 사용해 목표 포즈(target_pos, target_quat)를 추종하는 IK 연산
        :param target_pos: [x, y, z] 형태의 목표 월드 좌표 (NumPy 배열)
        :param target_quat: [x, y, z, w] 형태의 목표 월드 쿼터니언 (NumPy 배열)
        """
        if self.pump_index == -1: return None
        # 1. 현재 로봇의 9개 DOF 상태 가져오기
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_POS)
        current_q = np.array([state[0] for state in dof_states], dtype=np.float32)
        
        # 앞의 7개 관절만 최적화 대상으로 분리
        q_7dof = current_q[:7].copy()
        for item in range(max_iters):
            # 현재 엔드이펙터의 실시간 포즈 획득
            ee_pos, ee_rot = self.get_end_effector_pose()
            if ee_pos is None: break
            curr_p = np.array([ee_pos['x'], ee_pos['y'], ee_pos['z']])
            curr_q = R.from_quat([ee_rot['x'], ee_rot['y'], ee_rot['z'], ee_rot['w']])
            # 2. 위치 오차(Position Error) 계산
            pos_err = target_pos - curr_p
            # 3. 회전 오차(Rotation Error) 계산 (회전 벡터 변환)
            rot_err_obj = R.from_quat(target_quat) * curr_q.inv()
            rot_err = rot_err_obj.as_rotvec()
            # 6차원 작업 공간 오차 벡터 통합 (행벡터)
            twist_err = np.concatenate([pos_err, rot_err])
            # 수렴 조건 만족 시 탈출
            if np.linalg.norm(twist_err) < tol:
                break
            # 4. 수치적 수렴을 위한 가상 Jacobian(기울기) 업데이트 매핑
            # (원래는 J_pseudo * twist_err 이나, 수치적 근사를 위해 간단한 P-Gain 제어 형태로 변위 유도)
            # ⚠️ 수치적 정밀 역행렬 처리를 위해선 각 조인트 미소 변화에 따른 
            # 엔드이펙터의 변위 수치미분(Numerical Differentiation)을 루프 내에 구현해야 합니다.
            
            # 여기서는 부드러운 수렴을 위한 7자유도 의사역행렬 근사 가중치 주입 예시
            # 수치 미분용 가상 자코비안 Matrix 생성 (6 x 7)
            J = np.zeros((6, 7))
            epsilon = 1e-4
            
            for i in range(7):
                # i번째 관절을 아주 미세하게 꺾어봄
                q_eps = q_7dof.copy()
                q_eps[i] += epsilon
                
                # 임시로 시뮬레이션 타겟을 변경하여 포즈 변화 관찰 (또는 순운동학 Kinematics 수식 대입)
                # 수치 미분을 위해 7개 조인트를 가상으로 포워드 킹하는 가벼운 기하학적 헬퍼가 필요합니다.
            # 5. 계산된 조인트 타겟 변위를 기존 보간 버퍼에 반영
            # 여기서는 변위 업데이트 간략화 매핑 (실제 적용 시 수치 미분 J_pinv 사용 권장)
            # q_7dof += J_pinv @ twist_err * dt
            
        # 6. 최종 연산된 7개 관절 각도와 기존 8, 9번째 관절 각도를 병합하여 최종 버퍼 갱신
        final_targets = current_q.copy()
        final_targets[:7] = q_7dof
        
        return final_targets
    
    def get_end_effector_pose_relative_to_base(self):
        """
        로봇 베이스 링크(panda_link0)를 원점(0,0,0)으로 삼았을 때,
        엔드이펙터(cobot_pump) 본체의 상대적 위치 [x, y, z] 및 
        Rot Z-Y-X 순서의 각도 오프셋 [RotZ, RotY, RotX] (Degree)를 반환합니다.
        
        :return: (relative_pos, rpy_zyx_deg)
                 - relative_pos: [x, y, z] 형태의 위치 NumPy 배열 (미터 단위)
                 - rpy_zyx_deg: [Rot Z, Rot Y, Rot X] 형태의 각도 NumPy 배열 (도 단위)
        """
        if self.pump_index == -1 or self.base_index == -1:
            print("[에러] 링크 인덱스가 유효하지 않아 상대 포즈를 계산할 수 없습니다.")
            return None, None

        # 1. 현재 모든 링크의 실시간 월드 포즈(World Frame) 가져오기
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_POS)
        
        base_pose = body_states['pose'][self.base_index]
        ee_pose = body_states['pose'][self.pump_index]

        # 2. 베이스와 엔드이펙터의 월드 위치/회전 구조체 파싱
        base_p = np.array([base_pose['p']['x'], base_pose['p']['y'], base_pose['p']['z']])
        base_q = [base_pose['r']['x'], base_pose['r']['y'], base_pose['r']['z'], base_pose['r']['w']]
        base_rotation = R.from_quat(base_q)

        ee_p = np.array([ee_pose['p']['x'], ee_pose['p']['y'], ee_pose['p']['z']])
        ee_q = [ee_pose['r']['x'], ee_pose['r']['y'], ee_pose['r']['z'], ee_pose['r']['w']]
        ee_rotation = R.from_quat(ee_q)

        # 3. 🎯 상대 위치 역산 (베이스 로컬 공간 기준 변위)
        relative_pos = base_rotation.inv().apply(ee_p - base_p)

        # 4. 🎯 상대 회전 역산 (R_relative = R_base_inv * R_ee)
        relative_rotation = base_rotation.inv() * ee_rotation
        
        # 💡 [핵심 변환] SciPy의 'zyx' 대문자 문자열은 고유 회전(Intrinsic) 기준 
        # Z축 회전 -> Y축 회전 -> X축 회전 순으로 오일러 각을 추출합니다.
        # degrees=True 옵션을 주면 라디안에서 도(°) 단위로 자동 변환됩니다.
        rpy_zyx_deg = relative_rotation.as_euler('zyx', degrees=True)

        return relative_pos, rpy_zyx_deg
    
    def _compute_quintic_spline(self, q_start, q_target, t, t_max):
        """
        5차 다항식을 이용해 시작(t=0)과 끝(t=t_max)에서의 
        속도와 가속도가 0이 되는 부드러운 관절 각도를 계산합니다.
        """
        if t >= t_max:
            return q_target
        if t <= 0:
            return q_start
            
        # 정규화된 시간 (0 ~ 1)
        tau = t / t_max
        
        # 5차 다항식 보간 계수 연산 (Minimum Jerk Trajectory)
        # s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        
        return q_start + (q_target - q_start) * s