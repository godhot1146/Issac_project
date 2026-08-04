import numpy as np
from isaacgym import gymapi
import math

class LowProfileAMR:
    def __init__(self, gym, env, handle, planner=None):
        self.gym = gym
        self.env = env
        self.handle = handle
        self.planner = planner 
        
        # =================================================================
        # 🏎️ [공식 스펙 튜닝] Geek+ P800R 하드웨어 제원 동기화
        # =================================================================
        self.kp_linear = 5.0        # 142kg을 밀기 위한 부드러운 선속도 게인
        self.kp_angular = 8.0       # 무거운 기체를 확실하게 돌리기 위한 조향 게인
        self.max_lin_vel = 2.0      # ★ 공식 스펙 최고 속도 2.0m/s
        self.max_ang_vel = 1.5      # 안정적인 회전을 위한 각속도 제한
        
        self.lookahead_dist = 0.28 
        self.min_dist = 0.05       

        self.current_path = []
        self.current_wp_idx = 0
        
        # ★ 하강 시 물리 엔진 반발력을 억제하기 위한 마이너스 텐션 적용
        self.lift_target_pos = -0.05

        # 가속도 제한용 변수 및 하드웨어 치수 (기구학 연산용)
        self.prev_v_lin = 0.0
        self.prev_v_ang = 0.0
        self.track_width = 0.88     # 좌우 바퀴 간격 (0.44 * 2)
        self.wheel_radius = 0.08    # 바퀴 반지름 8cm

        self.is_docking_phase = True

        self.wheel_names = ["fl_joint", "fr_joint", "rl_joint", "rr_joint"]
        self.lift_name = "lift_joint"

        self._setup_physics()

    def _setup_physics(self):
        s_props = self.gym.get_actor_rigid_shape_properties(self.env, self.handle)
        for s in s_props:
            s.friction = 2.5            
            s.rolling_friction = 0.1
        self.gym.set_actor_rigid_shape_properties(self.env, self.handle, s_props)

        d_props = self.gym.get_actor_dof_properties(self.env, self.handle)
        
        for name in self.wheel_names:
            idx = self.find_joint_idx(name)
            if idx != -1:
                d_props["driveMode"][idx] = gymapi.DOF_MODE_VEL
                d_props["stiffness"][idx] = 0.0
                d_props["damping"][idx] = 500.0   # 142kg을 감당하기 위해 제동 댐핑 강화
                d_props["armature"][idx] = 0.05
            
        lift_idx = self.find_joint_idx(self.lift_name)
        if lift_idx != -1:
            d_props["driveMode"][lift_idx] = gymapi.DOF_MODE_POS
            d_props["stiffness"][lift_idx] = 500000.0  
            d_props["damping"][lift_idx] = 50000.0     
            d_props["velocity"][lift_idx] = 0.02      
            d_props["armature"][lift_idx] = 0.5
        
        self.gym.set_actor_dof_properties(self.env, self.handle, d_props)

    def find_joint_idx(self, name):
        try:
            return self.gym.find_actor_dof_index(self.env, self.handle, name, gymapi.DOMAIN_ACTOR)
        except:
            return -1

    def get_current_pose(self):
        c_states = self.gym.get_actor_rigid_body_states(self.env, self.handle, gymapi.STATE_ALL)
        curr_p, curr_r = c_states['pose']['p'][0], c_states['pose']['r'][0]
        return curr_p, curr_r

    def set_lift(self, command):
        if command == "UP":
            # ★ 공식 스펙 최대 인양 높이 55mm (0.055m)
            self.lift_target_pos = 0.055  
        elif command == "DOWN":
            self.lift_target_pos = -0.05   
            
    def set_target(self, curr_pos, target_pos):
        if self.planner:
            raw_path = self.planner.plan_path(curr_pos, target_pos, robot_name="low_amr")
            if raw_path and len(raw_path) > 1:
                self.current_path = raw_path[1:]
            else:
                self.current_path = raw_path
            self.current_wp_idx = 0

    def drive_to_target(self, curr_p, curr_r):
        num_dofs = self.gym.get_actor_dof_count(self.env, self.handle)
        vel_targets = np.zeros(num_dofs, dtype=np.float32)
        pos_targets = np.zeros(num_dofs, dtype=np.float32)
        
        lift_idx = self.find_joint_idx(self.lift_name)
        if lift_idx != -1:
            pos_targets[lift_idx] = self.lift_target_pos

        if not self.current_path:
            self.gym.set_actor_dof_velocity_targets(self.env, self.handle, vel_targets)
            self.gym.set_actor_dof_position_targets(self.env, self.handle, pos_targets)
            return True  

        final_wp = self.current_path[-1]
        dist_error = np.hypot(final_wp[0] - curr_p['x'], final_wp[1] - curr_p['y'])
        
        if dist_error < self.min_dist:
            self.prev_v_lin = 0.0
            self.prev_v_ang = 0.0
            self.gym.set_actor_dof_velocity_targets(self.env, self.handle, vel_targets)
            self.gym.set_actor_dof_position_targets(self.env, self.handle, pos_targets)
            self.current_path = []
            return True  

        target_wp = final_wp
        for i in range(self.current_wp_idx, len(self.current_path)):
            wp = self.current_path[i]
            dist_to_wp = np.hypot(wp[0] - curr_p['x'], wp[1] - curr_p['y'])
            if dist_to_wp >= self.lookahead_dist:
                target_wp = wp
                self.current_wp_idx = i 
                break

        siny_cosp = 2.0 * (curr_r['w'] * curr_r['z'] + curr_r['x'] * curr_r['y'])
        cosy_cosp = 1.0 - 2.0 * (curr_r['y']**2 + curr_r['z']**2)
        curr_yaw = np.arctan2(siny_cosp, cosy_cosp)

        err_x = target_wp[0] - curr_p['x']
        if self.is_docking_phase:
            err_y = 0.77 - curr_p['y']  
        else:
            err_y = target_wp[1] - curr_p['y']  
        
        current_wp_dist = np.hypot(err_x, err_y)

        if not self.is_docking_phase:
            target_yaw = np.arctan2(err_y, err_x) + np.pi
            yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))
            
            if abs(yaw_error) > 0.2:
                v_lin = -0.3  
            else:
                v_lin = np.clip(-self.kp_linear * current_wp_dist * np.cos(yaw_error), -self.max_lin_vel, self.max_lin_vel)
                
            v_ang = np.clip(self.kp_angular * yaw_error, -self.max_ang_vel, self.max_ang_vel)
        else:
            target_yaw = np.arctan2(err_y, err_x)
            yaw_error = np.arctan2(np.sin(target_yaw - curr_yaw), np.cos(target_yaw - curr_yaw))
            
            if abs(yaw_error) > 0.2:
                v_lin = 0.3  
            else:
                v_lin = np.clip(self.kp_linear * current_wp_dist * np.cos(yaw_error), -self.max_lin_vel, self.max_lin_vel)
                
            v_ang = np.clip(self.kp_angular * yaw_error, -self.max_ang_vel, self.max_ang_vel)

        MAX_ACCEL = 0.04  
        MAX_ALPHA = 0.06  
        
        v_lin_error = v_lin - self.prev_v_lin
        v_lin = self.prev_v_lin + np.clip(v_lin_error, -MAX_ACCEL, MAX_ACCEL)
        self.prev_v_lin = v_lin
        
        v_ang_error = v_ang - self.prev_v_ang
        v_ang = self.prev_v_ang + np.clip(v_ang_error, -MAX_ALPHA, MAX_ALPHA)
        self.prev_v_ang = v_ang

        # ★ 차분 구동 기구학 수식 적용 (선속도/각속도를 실제 바퀴 회전 rad/s로 변환)
        v_left_linear = v_lin - v_ang * (self.track_width / 2.0)
        v_right_linear = v_lin + v_ang * (self.track_width / 2.0)

        ls = v_left_linear / self.wheel_radius
        rs = v_right_linear / self.wheel_radius

        fl_idx = self.find_joint_idx("fl_joint")
        fr_idx = self.find_joint_idx("fr_joint")
        rl_idx = self.find_joint_idx("rl_joint")
        rr_idx = self.find_joint_idx("rr_joint")
        
        if fl_idx != -1: vel_targets[fl_idx] = ls
        if rl_idx != -1: vel_targets[rl_idx] = ls
        if fr_idx != -1: vel_targets[fr_idx] = rs
        if rr_idx != -1: vel_targets[rr_idx] = rs
        
        self.gym.set_actor_dof_velocity_targets(self.env, self.handle, vel_targets)
        self.gym.set_actor_dof_position_targets(self.env, self.handle, pos_targets)

        return False