import numpy as np
from isaacgym import gymapi

class LiftingAMR:
    def __init__(self, gym, sim, env, handle):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.handle = handle

        # -----------------------------------------------------------------
        # 1. 하드웨어 세원 및 제어 파라미터 (상수 설정)
        # -----------------------------------------------------------------
        self.TRACK_WIDTH = 0.80     # 좌우 구동륜 바퀴 사이의 거리 (m)
        self.WHEEL_RADIUS = 0.15    # 구동 바퀴 반지름 (m)
        
        # 주행 속도 및 가속도 소프트웨어 한계 한도 세팅 (마스트 떨림 방지)
        self.MAX_LINEAR_VEL = 1.5   # 최대 선속도 (m/s)
        self.MAX_ANGULAR_VEL = 3.5  # 최대 각속도 (rad/s)
        self.MAX_LINEAR_ACCEL = 0.04 # 가속도 제한 필터 (프레임당 최대 속도 변화량)
        self.MAX_ANGULAR_ACCEL = 0.4

        self.Kp_LIN = 2.5           # 선속도 제어 비례 게인
        self.Kp_ANG = 6.0           # 조향 각속도 제어 비례 게인
        self.ARRIVE_THRESHOLD = 0.01 # 목적지 도달 인정 공차 (5cm)

        self.LIFT_JOINT_NAME = "lift_z_joint"
        self.LIFT_MIN_HEIGHT = 0.0
        self.LIFT_MAX_HEIGHT = 2.5

        self.ROTATION_JOINT_NAME = "lift_rotation_joint"

        self.FORK_JOINT_NAME = "fork_extension_joint"
        self.FORK_MIN_LIMIT = 0.0
        self.FORK_MAX_LIMIT = 1.2  # URDF 상 upper="0.7" 반영

        self.LEFT_CLAW_JOINT_NAME = "left_claw_joint"
        self.RIGHT_CLAW_JOINT_NAME = "right_claw_joint"
        self.CLAW_MIN_LIMIT = -0.8  # URDF lower
        self.CLAW_MAX_LIMIT = 0.8   # URDF upper

        # URDF 매핑 네이밍 정보
        self.LEFT_JOINT_NAME = "left_wheel_joint"
        self.RIGHT_JOINT_NAME = "right_wheel_joint"
        self.CASTER_LINK_NAMES = ["cast_wheel_link1", "cast_wheel_link2", "cast_wheel_link3", "cast_wheel_link4"]
        self.DRIVE_LINK_NAMES = ["left_wheel_link", "right_wheel_link"]

        # -----------------------------------------------------------------
        # 2. 내부 상태 변수 초기화
        # -----------------------------------------------------------------
        self.current_v_lin = 0.0
        self.current_v_ang = 0.0
        
        self.left_wheel_idx = -1
        self.right_wheel_idx = -1
        self.lift_joint_idx = -1
        self.rot_joint_idx = -1
        self.dof_count = 0

        self.grid_move_stage = 0   # 0: X축 맞추기 단계, 1: Y축 맞추기 단계
        self.target_yaw_fixed = None # 회전 시작 시 고정할 목표 각도
        self.turn_complete = False
        self.locked_turn_direction = None

        # 초기 물리 가속 및 접지력 튜닝 자동 셋업
        self._cache_joint_indices()
        self.configure_actuators(damping=800.0, max_effort=500000.0)
        self.configure_contact_surface(caster_friction=0.0, drive_friction=2.5)

        self.dof_position_targets = np.zeros(self.dof_count, dtype=np.float32)
        self.dof_velocity_targets = np.zeros(self.dof_count, dtype=np.float32)

    def apply_actuator_commands(self):
        """매 스텝 축적된 포지션/벨로시티 버퍼 명령을 Isaac Gym 제어기에 단 한 번만 주입합니다."""
        self.gym.set_actor_dof_position_targets(self.env, self.handle, self.dof_position_targets)
        self.gym.set_actor_dof_velocity_targets(self.env, self.handle, self.dof_velocity_targets)

    def _cache_joint_indices(self):
        """조인트 인덱스 조회를 초기 1회만 실행하여 런타임 오버헤드를 방지합니다."""
        self.dof_count = self.gym.get_actor_dof_count(self.env, self.handle)
        self.left_wheel_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LEFT_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.right_wheel_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.RIGHT_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.lift_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LIFT_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.rot_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.ROTATION_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.fork_joint_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.FORK_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.left_claw_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.LEFT_CLAW_JOINT_NAME, gymapi.DOMAIN_ACTOR)
        self.right_claw_idx = self.gym.find_actor_dof_index(self.env, self.handle, self.RIGHT_CLAW_JOINT_NAME, gymapi.DOMAIN_ACTOR)

        if self.left_wheel_idx == -1 or self.right_wheel_idx == -1:
            print("[경고] URDF 내 구동륜 조인트 이름을 찾을 수 없습니다.")

    def configure_actuators(self, damping=800.0, max_effort=500000.0):
        """모터의 한계 토크 해제 및 떨림 방지를 위한 제어 프로퍼티를 일괄 셋업합니다."""
        dof_props = self.gym.get_actor_of_properties(self.env, self.handle) if hasattr(self.gym, 'get_actor_of_properties') else self.gym.get_actor_dof_properties(self.env, self.handle)
        
        # 전 조인트 속도 제어 모드로 초기화
        dof_props['driveMode'].fill(gymapi.DOF_MODE_VEL)
        dof_props['stiffness'].fill(0.0)
        dof_props['damping'].fill(damping)

        # 구동륜 모터 파워 한계 강화
        for idx in [self.left_wheel_idx, self.right_wheel_idx]:
            if idx != -1:
                dof_props['effort'][idx] = max_effort
                dof_props['velocity'][idx] = self.MAX_ANGULAR_VEL / self.WHEEL_RADIUS * 2.0

        if self.lift_joint_idx != -1:
            dof_props['driveMode'][self.lift_joint_idx] = gymapi.DOF_MODE_POS
            # 고중량 셔틀이 처지지 않게 강한 강성(stiffness)과 댐핑을 부여합니다.
            dof_props['stiffness'][self.lift_joint_idx] = 5000.0  
            dof_props['damping'][self.lift_joint_idx] = 1200.0    
            dof_props['effort'][self.lift_joint_idx] = 5000.0       # URDF 내의 effort="500" 매핑

        if self.rot_joint_idx != -1:
            dof_props['driveMode'][self.rot_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.rot_joint_idx] = 8000.0   # 회전 모터 고유 강성
            dof_props['damping'][self.rot_joint_idx] = 400.0      # 떨림 방지 감쇠
            dof_props['effort'][self.rot_joint_idx] = 100.0       # URDF 내의 effort="100"

        if self.fork_joint_idx != -1:
            dof_props['driveMode'][self.fork_joint_idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][self.fork_joint_idx] = 10000.0  # 전진/후진 시 처짐 및 밀림 방지 강성
            dof_props['damping'][self.fork_joint_idx] = 500.0      # 오버슈트 방지 댐핑
            dof_props['effort'][self.fork_joint_idx] = 3000.0       # URDF 내의 effort="150" 매핑

        for idx in [self.left_claw_idx, self.right_claw_idx]:
            if idx != -1:
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][idx] = 5000.0   # 물체를 단단히 고정하기 위한 강성
                dof_props['damping'][idx] = 200.0     # 채터링(떨림) 방지 댐핑
                dof_props['effort'][idx] = 50.0       # URDF 내의 effort="50" 매핑

        self.gym.set_actor_dof_properties(self.env, self.handle, dof_props)

    def configure_contact_surface(self, caster_friction=0.0, drive_friction=2.5):
        """IndexRange 비대칭 예외를 회피하여 캐스터와 구동륜 마찰력을 분리 제어합니다."""
        shape_props = self.gym.get_actor_rigid_shape_properties(self.env, self.handle)
        num_bodies = self.gym.get_actor_rigid_body_count(self.env, self.handle)
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.env, self.handle)

        for body_idx in range(num_bodies):
            body_name = rigid_body_names[body_idx]
            shape_range = self.gym.get_actor_rigid_body_shape_indices(self.env, self.handle)[body_idx]
            
            for shape_idx in range(shape_range.start, shape_range.start + shape_range.count):
                if body_name in self.CASTER_LINK_NAMES:
                    shape_props[shape_idx].friction = caster_friction
                    shape_props[shape_idx].rolling_friction = 0.0
                elif body_name in self.DRIVE_LINK_NAMES:
                    shape_props[shape_idx].friction = drive_friction
                    shape_props[shape_idx].rolling_friction = 0.1

        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, shape_props)

    def set_claw_grip(self, target_angle):
        """
        양쪽 클로의 목표 각도(라디안)를 인가합니다. (-0.7854 ~ 0.7854 사이)
        하나의 입력값으로 좌우 대칭 구동 버퍼를 동시에 업데이트합니다.
        """
        safe_angle = np.clip(target_angle, self.CLAW_MIN_LIMIT, self.CLAW_MAX_LIMIT)
        
        # 💡 URDF 디자인에 따라 한쪽 클로가 반대로 움직여야 대칭으로 다물어질 수 있습니다.
        # 만약 시뮬레이션에서 양 클로가 같은 방향으로 회전(한쪽은 열리고 한쪽은 닫힘)한다면,
        # 아래 right_claw_idx 부분의 safe_angle 앞에 마이너스(-) 부호를 붙여주세요.
        if self.left_claw_idx != -1:
            self.dof_position_targets[self.left_claw_idx] = safe_angle
            
        if self.right_claw_idx != -1:
            self.dof_position_targets[self.right_claw_idx] = safe_angle

    def get_claw_positions(self):
        """현재 좌우 클로 조인트의 실제 포지션 각도(라디안)를 반환합니다."""
        if self.left_claw_idx == -1 or self.right_claw_idx == -1:
            return 0.0, 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.left_claw_idx], dof_states['pos'][self.right_claw_idx]

    def set_fork_extension(self, target_length):
        """
        포크 확장 슬라이더의 목표 길이(m)를 인가합니다. (0.0m ~ 0.7m 사이)
        클래스 전역 공유 포시전 버퍼의 포크 조인트 슬롯 값만 정밀 교체합니다.
        """
        if self.fork_joint_idx != -1:
            # URDF 소프트웨어 한계 임계값 클리핑 (0.0m ~ 0.7m 초과 방지)
            safe_length = np.clip(target_length, self.FORK_MIN_LIMIT, self.FORK_MAX_LIMIT)
            
            # 공유 버퍼 슬롯 업데이트 (리프트, 회전 등 타 슬롯 데이터 보존)
            self.dof_position_targets[self.fork_joint_idx] = safe_length

    def get_fork_extension(self):
        """현재 포크 확장 조인트의 실제 연장 길이(m)를 반환합니다."""
        if self.fork_joint_idx == -1:
            return 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.fork_joint_idx]

    def set_shuttle_rotation(self, target_radian):
        """셔틀 회전 목표 각도를 공유 버퍼에 안전하게 업데이트합니다."""
        if self.rot_joint_idx != -1:
            safe_angle = np.clip(target_radian, -1.5708, 1.5708)
            
            # 🎯 [수정] 마찬가지로 다른 조인트(리프트, 바퀴) 데이터를 건드리지 않고
            # 회전 조인트 슬롯 값만 덮어씁니다.
            self.dof_position_targets[self.rot_joint_idx] = safe_angle

    def get_shuttle_rotation(self):
        """현재 셔틀 회전 조인트의 실제 포지션 각도(라디안)를 반환합니다."""
        if self.rot_joint_idx == -1:
            return 0.0
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.rot_joint_idx]

    def get_lift_height(self):
        """현재 리프팅 조인트의 실제 포지션(높이, m)을 반환합니다."""
        if self.lift_joint_idx == -1:
            return 0.0
        # 전체 DOF 상태 리스트를 가져와 리프팅 조인트 인덱스의 위치값만 파싱
        dof_states = self.gym.get_actor_dof_states(self.env, self.handle, gymapi.STATE_ALL)
        return dof_states['pos'][self.lift_joint_idx]

    def get_pose(self):
        """로봇의 전역 위치 좌표(p)와 회전 쿼터니언(r)을 직관적으로 반환합니다."""
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        return body_states['pose']['p'][0], body_states['pose']['r'][0]

    def _get_current_yaw(self, q):
        """쿼터니언 자세 값으로부터 라디안 단위의 Yaw 회전각(Z축 회전)을 계산합니다."""
        siny_cosp = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
        cosy_cosp = 1.0 - 2.0 * (q['y']**2 + q['z']**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def turn_to_yaw(self, target_yaw):
        """
        시작할 때 최단 거리 방향을 딱 한 번만 '고정(Lock)'하여 
        회전 도중 실시간으로 방향이 바뀌며 떨리는 현상을 완벽히 차단한 회전 함수.
        """
        _, curr_r = self.get_pose()
        curr_yaw = self._get_current_yaw(curr_r)

        ANGLE_ARRIVE_THRESHOLD = 0.01  

        # 1. 매 스텝 현재 각도 오차 계산
        yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))

        # 데드존 안으로 들어오면 성공 플래그와 함께 방향 고정 해제 후 정지
        if abs(yaw_error) < ANGLE_ARRIVE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            self.locked_turn_direction = None  # 방향 락 해제
            return True

        # 2. [방향 고정 로직] 회전을 처음 시작할 때 딱 한 번만 방향을 결정합니다.
        if self.locked_turn_direction is None:
            # yaw_error가 양수면 좌회전(1), 음수면 우회전(-1)으로 방향을 박아버립니다.
            self.locked_turn_direction = 1.0 if yaw_error < 0 else -1.0

        # 3. 실시간 yaw_error의 부호는 무시하고, 오직 오차 크기(abs)와 고정된 방향만 사용합니다.
        p_control_speed = self.Kp_ANG * abs(yaw_error)
        scaled_turn_speed = np.clip(p_control_speed, 0.6, self.MAX_ANGULAR_VEL)
        angular_cmd = self.locked_turn_direction * scaled_turn_speed
        
        self.set_twist_velocity(linear_x=0.0, angular_z=angular_cmd)
        return False

    def drive_along_axis(self, axis, target_coordinate, direction="FORWARD"):
        """
        지정된 축을 따라 정해진 방향(FORWARD/BACKWARD)으로 직선 주행하는 독립 함수.
        axis: 'X' 또는 'Y'
        target_coordinate: 도달하고자 하는 목표 축의 절대 좌표값 (float)
        direction: 'FORWARD' (전진, +속도) 또는 'BACKWARD' (후진, -속도)
        목표 축 좌표에 도달 시 True, 주행 중일 시 False를 반환합니다.
        """
        curr_p, _ = self.get_pose()
        DISTANCE_THRESHOLD = 0.01 # 축 도달 인정 공차 (5cm)

        # 1. 제어하고자 하는 축의 현재 위치 및 오차 바인딩
        current_val = curr_p['x'] if axis.upper() == 'X' else curr_p['y']
        axis_error = target_coordinate - current_val

        # 목적지 도달 시 완전 정지 후 완료 시그널 리턴
        if abs(axis_error) < DISTANCE_THRESHOLD:
            self.set_twist_velocity(0.0, 0.0)
            return True

        # 2. 전진/후진 선택에 따른 선속도 부호 강제 정의
        # 오차 크기에 비례하여 감속하되 가속도 필터로 토스합니다.
        base_speed = np.clip(self.Kp_LIN * abs(axis_error), 0.6, self.MAX_LINEAR_VEL)
        
        if direction.upper() == "FORWARD":
            linear_cmd = base_speed      # 정방향 전진 (+)
        elif direction.upper() == "BACKWARD":
            linear_cmd = -base_speed     # 역방향 후진 (-)
        else:
            linear_cmd = base_speed

        # 직선성을 보장하기 위해 조향 각속도는 완벽하게 0.0 고정
        self.set_twist_velocity(linear_x=linear_cmd, angular_z=0.0)
        return False

    def move_to_point(self, target_x, target_y, final_direction='+x'):
        """
        [방향 문자열 적용판]
        Stage 0: X축 정렬 및 주행
        Stage 1: Y축 정렬 및 주행
        Stage 2: 주행 종료 후 final_direction('+x', '-x', '+y', '-y') 방향으로 최종 제자리 정렬
        
        :param final_direction: '+x', '-x', '+y', '-y' (대소문자 무관)
        """
        curr_p, _ = self.get_pose()
        DISTANCE_THRESHOLD = 0.05

        # [Stage 0: X축 정렬 및 주행]
        if self.grid_move_stage == 0:
            err_x = target_x - curr_p['x']
            if abs(err_x) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None 
                self.turn_complete = False  
                self.grid_move_stage = 1     
                return False

            if self.target_yaw_fixed is None:
                self.target_yaw_fixed = 0.0 if err_x > 0 else np.pi
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='X', target_coordinate=target_x, direction="FORWARD")

        # [Stage 1: Y축 정렬 및 주행]
        elif self.grid_move_stage == 1:
            err_y = target_y - curr_p['y']
            if abs(err_y) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 2 
                return False

            if self.target_yaw_fixed is None:
                self.target_yaw_fixed = np.pi / 2.0 if err_y > 0 else -np.pi / 2.0
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='Y', target_coordinate=target_y, direction="FORWARD")

        # [Stage 2: 지정된 방향 문자열을 기반으로 최종 Yaw 제정렬]
        elif self.grid_move_stage == 2:
            # 💡 방향 문자열 매핑 딕셔너리 정의
            direction_map = {
                '+x': 0.0,
                '+y': np.pi / 2.0,
                '-x': np.pi,
                '-y': -np.pi / 2.0
            }
            
            # 입력받은 문자열 정제 및 예외 처리 (정의되지 않은 값일 경우 기본값 +x 축 정렬)
            clean_dir = str(final_direction).strip().lower()
            calculated_final_yaw = direction_map.get(clean_dir, 0.0)
            
            # 계산된 라디안 각도로 제자리 회전 지시
            if self.turn_to_yaw(calculated_final_yaw):
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 0 
                print(f"[주행 완료] 최종 지정 방향 '{clean_dir}' ({calculated_final_yaw} rad) 정렬 완료.")
                return True 

        return False
    
    def move_to_point(self, target_x, target_y, backward=False):
        """
        [후진 기구학 수치 오염 완치판]
        X/Y 직교 순차 이동 마스터 함수.
        Stage 0 내부의 중복 오프셋 조건문을 제거하여 후진 시의 반동과 단선 주행을 완벽히 해결합니다.
        """
        curr_p, _ = self.get_pose()
        DISTANCE_THRESHOLD = 0.01
        
        # 주행 기구학 방향 세팅
        move_dir = "BACKWARD" if backward else "FORWARD"

        # -----------------------------------------------------------------
        # [Stage 0: X축 거리 맞추기 단계]
        # -----------------------------------------------------------------
        if self.grid_move_stage == 0:
            err_x = target_x - curr_p['x']
            
            if abs(err_x) < DISTANCE_THRESHOLD:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None 
                self.turn_complete = False  
                self.grid_move_stage = 1     
                return False

            if self.target_yaw_fixed is None:
                # 💡 최초 1회만 X축 회전 방향 정렬 각도를 깔끔하게 잠급니다.
                if backward:
                    self.target_yaw_fixed = np.pi if err_x > 0 else 0.0
                else:
                    self.target_yaw_fixed = 0.0 if err_x > 0 else np.pi
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='X', target_coordinate=target_x, direction=move_dir)

        # -----------------------------------------------------------------
        # [Stage 1: Y축 거리 맞추기 단계]
        # -----------------------------------------------------------------
        elif self.grid_move_stage == 1:
            err_y = target_y - curr_p['y']

            print("err_y: ",err_y)
            
            #if abs(err_y) < DISTANCE_THRESHOLD:
            if abs(err_y) < 0.05:
                self.set_twist_velocity(0.0, 0.0)
                self.target_yaw_fixed = None
                self.turn_complete = False
                self.grid_move_stage = 0 
                return True # 최종 도달 신호

            if self.target_yaw_fixed is None:
                # 💡 최초 1회만 Y축 회전 방향 정렬 각도를 깔끔하게 잠급니다.
                if backward:
                    self.target_yaw_fixed = -np.pi / 2.0 if err_y > 0 else np.pi / 2.0
                else:
                    self.target_yaw_fixed = np.pi / 2.0 if err_y > 0 else -np.pi / 2.0
                self.turn_complete = False

            if not self.turn_complete:
                if self.turn_to_yaw(self.target_yaw_fixed):
                    self.turn_complete = True
            else:
                self.drive_along_axis(axis='Y', target_coordinate=target_y, direction=move_dir)

        return False
    
    def set_lift_height(self, target_height):
        """리프팅 목표 높이를 공유 버퍼에 안전하게 업데이트합니다."""
        if self.lift_joint_idx != -1:
            safe_height = np.clip(target_height, self.LIFT_MIN_HEIGHT, self.LIFT_MAX_HEIGHT)
            
            # 🎯 [수정] 배열을 새로 만들지 않고, 클래스가 들고 있는 전역 공유 버퍼의 
            # 리프트 조인트 슬롯 값만 정밀 타겟으로 교체합니다. (타 슬롯 데이터 보존)
            self.dof_position_targets[self.lift_joint_idx] = safe_height

    def set_twist_velocity(self, linear_x, angular_z):
        """
        ROS 표준 Twist 스타일 명령을 받아 가속도 필터링 후 
        클래스 전역 속도 버퍼(self.dof_velocity_targets)에 안전하게 업데이트합니다.
        """
        # 1. 입력 명령 한도 클리핑
        target_v_lin = np.clip(linear_x, -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        target_v_ang = np.clip(angular_z, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)

        # 2. 슬루 레이트(가속도 제한) 필터링
        v_lin_err = target_v_lin - self.current_v_lin
        self.current_v_lin += np.clip(v_lin_err, -self.MAX_LINEAR_ACCEL, self.MAX_LINEAR_ACCEL)

        v_ang_err = target_v_ang - self.current_v_ang
        self.current_v_ang += np.clip(v_ang_err, -self.MAX_ANGULAR_ACCEL, self.MAX_ANGULAR_ACCEL)

        # 3. 차분 구동형 휠 기구학 수식 전개
        v_left_linear = self.current_v_lin - self.current_v_ang * (self.TRACK_WIDTH / 2.0)
        v_right_linear = self.current_v_lin + self.current_v_ang * (self.TRACK_WIDTH / 2.0)

        # 회전 각속도(rad/s) 단위로 리스케일링
        rad_sec_left = v_left_linear / self.WHEEL_RADIUS
        rad_sec_right = v_right_linear / self.WHEEL_RADIUS

        # 4. 🎯 [수정] 배열을 새로 생성하지 않고 구동륜 슬롯만 업데이트
        # 타 위치 제어 조인트들의 벨로시티 슬롯은 건드리지 않음 (기본값 0이 아닌 엔진 자율에 맡김)
        if self.left_wheel_idx != -1:   
            self.dof_velocity_targets[self.left_wheel_idx] = rad_sec_left
        if self.right_wheel_idx != -1:  
            self.dof_velocity_targets[self.right_wheel_idx] = rad_sec_right