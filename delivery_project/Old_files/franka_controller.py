import numpy as np

class FrankaController:
    def __init__(self, gym, env, handle):
        self.gym = gym
        self.env = env
        self.handle = handle
        
        # 동작 시퀀스 정의
        self.states = {
            "ready": [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04],
            "pick": [0, 0.8, 0, -1.75, 0, 2.50, 0.90, 0.04, 0.04],
            "close": [0, 0.8, 0, -1.75, 0, 2.50, 0.90, 0.0, 0.0],
            "lift": [0, -0.5, 0, -2.0, 0, 1.50, 0.90, 0.0, 0.0],
            "rotate": [1.571, -0.5, 0, -2.0, 0, 2.50, 0.90, 0.0, 0.0],
            "place": [1.571, 0.35, 0, -1.45, 0, 2.50, 0.6, 0.0, 0.0],
            "release": [1.571, 0.35, 0, -1.45, 0, 2.50, 0.6, 0.04, 0.04],
            "lift_up": [1.571, -0.5, 0, -1.45, 0, 1.50, 0.6, 0.04, 0.04]
        }
        self.done = False

    def update(self, frame_count):
        if self.done:
            return False # 작업 완료

        step = (frame_count // 100)
        is_loaded_signal = False
        
        if step == 0: targets = self.states["ready"]
        elif step == 1: targets = self.states["pick"]
        elif step == 2: targets = self.states["close"]
        elif step == 3: targets = self.states["lift"]
        elif step == 4: targets = self.states["rotate"]
        elif step == 5: targets = self.states["place"]
        elif step == 6: targets = self.states["release"]
        elif step == 7: 
            targets = self.states["lift_up"]
            is_loaded_signal = True # 적재 완료 신호 발생
        else:
            targets = self.states["ready"]
            self.done = True
            
        self.gym.set_actor_dof_position_targets(self.env, self.handle, np.array(targets, dtype=np.float32))
        return is_loaded_signal
