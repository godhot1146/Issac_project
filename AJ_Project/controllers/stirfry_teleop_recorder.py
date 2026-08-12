"""볶음 그릇 수동 파지 동작을 자동 시퀀스 분석용으로 기록한다.

매 프레임 원본 상태는 메모리에 보존하고, 출력할 때 같은 키/접촉 상태가
이어지는 프레임을 구간으로 압축한다. 출력된 TRACE 블록만 복사해도 TSC 목표,
실제 관절/그리퍼 자세, 그릇 운동, 접촉 및 상승 시점을 재구성할 수 있다.
"""
import json

import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation


class StirfryTeleopRecorder:
    FORMAT_VERSION = 1
    MAX_SEGMENT_FRAMES = 30
    LIFT_THRESHOLD_M = 0.015

    def __init__(
        self,
        gym,
        env,
        arm,
        bowl_actor,
        teleop,
        dt,
        gripper_body_name="stirfry_gripper_link",
    ):
        self.gym = gym
        self.env = env
        self.arm = arm
        self.bowl_actor = bowl_actor
        self.teleop = teleop
        self.dt = float(dt)
        self.samples = []
        self.active = True

        arm_body_names = gym.get_actor_rigid_body_names(env, arm.actor)
        if gripper_body_name not in arm_body_names:
            raise RuntimeError(f"rigid body not found: {gripper_body_name}")
        self.gripper_body_index = arm_body_names.index(gripper_body_name)
        self.gripper_env_body = gym.get_actor_rigid_body_index(
            env, arm.actor, self.gripper_body_index, gymapi.DOMAIN_ENV
        )
        self.bowl_env_body = gym.get_actor_rigid_body_index(
            env, bowl_actor, 0, gymapi.DOMAIN_ENV
        )

        bowl_state = self._body_state(bowl_actor, 0)
        self.initial_bowl_z = bowl_state["p"][2]
        print(
            "[수동 궤적 기록 시작] 자동 접근 인계 자세를 frame=0으로 기록합니다. "
            "성공 후 ENTER를 누르면 복사용 TRACE 블록이 출력됩니다.",
            flush=True,
        )
        self.record()

    @staticmethod
    def _vec3(value):
        return [float(value["x"]), float(value["y"]), float(value["z"])]

    @staticmethod
    def _quat(value):
        return [
            float(value["x"]),
            float(value["y"]),
            float(value["z"]),
            float(value["w"]),
        ]

    @staticmethod
    def _rounded(values, digits=6):
        return [round(float(value), digits) for value in values]

    def _body_state(self, actor, body_index):
        states = self.gym.get_actor_rigid_body_states(
            self.env, actor, gymapi.STATE_ALL
        )
        state = states[body_index]
        return {
            "p": self._vec3(state["pose"]["p"]),
            "q": self._quat(state["pose"]["r"]),
            "v": self._vec3(state["vel"]["linear"]),
            "w": self._vec3(state["vel"]["angular"]),
        }

    def _contact_state(self):
        count = 0
        impulse = 0.0
        for contact in self.gym.get_env_rigid_contacts(self.env):
            body0 = int(contact["body0"])
            body1 = int(contact["body1"])
            if {body0, body1} != {self.gripper_env_body, self.bowl_env_body}:
                continue
            count += 1
            names = getattr(getattr(contact, "dtype", None), "names", ()) or ()
            if "lambda" in names:
                impulse += abs(float(contact["lambda"]))
        return count, impulse

    def _commanded_joints(self):
        if self.teleop.mode == "jsc":
            return np.asarray(self.teleop.joint_target, dtype=np.float64)
        if self.teleop.mode == "tsc" and self.arm._last_q is not None:
            return np.asarray(self.arm._last_q, dtype=np.float64)
        return np.asarray(self.arm.current_joints(), dtype=np.float64)

    def record(self):
        """시뮬레이션 fetch_results 이후 호출한다."""
        if not self.active:
            return

        gripper = self._body_state(self.arm.actor, self.gripper_body_index)
        bowl = self._body_state(self.bowl_actor, 0)
        contact_count, contact_impulse = self._contact_state()

        tcp_p, tcp_R = self.arm.current_pose()
        target_q = Rotation.from_matrix(self.teleop.ori_R).as_quat()
        tcp_q = Rotation.from_matrix(tcp_R).as_quat()
        gripper_R = Rotation.from_quat(gripper["q"]).as_matrix()
        bowl_R = Rotation.from_quat(bowl["q"]).as_matrix()
        bowl_in_gripper_p = gripper_R.T @ (
            np.asarray(bowl["p"]) - np.asarray(gripper["p"])
        )
        bowl_in_gripper_q = Rotation.from_matrix(
            gripper_R.T @ bowl_R
        ).as_quat()
        lift_m = float(bowl["p"][2] - self.initial_bowl_z)

        self.samples.append(
            {
                "frame": len(self.samples),
                "mode": self.teleop.mode,
                "held": sorted(self.teleop.held),
                "target_p_base": self._rounded(self.teleop.cart_target),
                "target_q_base_xyzw": self._rounded(target_q),
                "q_cmd_rad": self._rounded(self._commanded_joints()),
                "q_actual_rad": self._rounded(self.arm.current_joints()),
                "tcp_actual_p_base": self._rounded(tcp_p),
                "tcp_actual_q_base_xyzw": self._rounded(tcp_q),
                "gripper_p_world": self._rounded(gripper["p"]),
                "gripper_q_world_xyzw": self._rounded(gripper["q"]),
                "bowl_p_world": self._rounded(bowl["p"]),
                "bowl_q_world_xyzw": self._rounded(bowl["q"]),
                "bowl_v_world": self._rounded(bowl["v"]),
                "bowl_w_world": self._rounded(bowl["w"]),
                "bowl_p_gripper": self._rounded(bowl_in_gripper_p),
                "bowl_q_gripper_xyzw": self._rounded(bowl_in_gripper_q),
                "contact_count": contact_count,
                "contact_impulse": round(contact_impulse, 6),
                "lift_m": round(lift_m, 6),
                "lifted": lift_m >= self.LIFT_THRESHOLD_M,
            }
        )

    @staticmethod
    def _signature(sample):
        return (
            sample["mode"],
            tuple(sample["held"]),
            sample["contact_count"] > 0,
            sample["lifted"],
        )

    @staticmethod
    def _endpoint(sample):
        keys = (
            "target_p_base",
            "target_q_base_xyzw",
            "q_cmd_rad",
            "q_actual_rad",
            "tcp_actual_p_base",
            "tcp_actual_q_base_xyzw",
            "gripper_p_world",
            "gripper_q_world_xyzw",
            "bowl_p_world",
            "bowl_q_world_xyzw",
            "bowl_v_world",
            "bowl_w_world",
            "bowl_p_gripper",
            "bowl_q_gripper_xyzw",
            "contact_count",
            "contact_impulse",
            "lift_m",
            "lifted",
        )
        return {key: sample[key] for key in keys}

    def _segments(self):
        if not self.samples:
            return []
        segments = []
        start = 0
        for index in range(1, len(self.samples)):
            signature_changed = (
                self._signature(self.samples[index])
                != self._signature(self.samples[index - 1])
            )
            span_full = index - start >= self.MAX_SEGMENT_FRAMES
            if signature_changed or span_full:
                end = index - 1
                segments.append((start, end))
                start = index
        if not segments or segments[-1][1] != len(self.samples) - 1:
            segments.append((start, len(self.samples) - 1))
        return segments

    def finish(self, result):
        """기록을 멈추고 터미널 복사용 구조화 TRACE 블록을 출력한다."""
        if not self.active:
            return
        self.active = False
        if not self.samples:
            return

        meta = {
            "type": "meta",
            "format": "stirfry_teleop_trace",
            "version": self.FORMAT_VERSION,
            "result": result,
            "dt_s": self.dt,
            "frames": len(self.samples),
            "duration_s": round((len(self.samples) - 1) * self.dt, 6),
            "cart_step_m_per_frame": float(self.teleop.cart_step),
            "ori_step_deg_per_frame": round(
                float(np.rad2deg(self.teleop.ori_step)), 6
            ),
            "joint_step_rad_per_frame": float(self.teleop.joint_step),
            "initial_bowl_z_world": round(self.initial_bowl_z, 6),
            "lift_threshold_m": self.LIFT_THRESHOLD_M,
            "quaternion_order": "xyzw",
        }
        print("\n===== STIRFRY_TELEOP_TRACE_BEGIN =====", flush=True)
        print("TRACE_META " + json.dumps(meta, separators=(",", ":")), flush=True)
        print(
            "TRACE_START "
            + json.dumps(self._endpoint(self.samples[0]), separators=(",", ":")),
            flush=True,
        )
        for sequence, (start, end) in enumerate(self._segments()):
            first = self.samples[start]
            last = self.samples[end]
            segment = {
                "type": "segment",
                "seq": sequence,
                "f0": first["frame"],
                "f1": last["frame"],
                "duration_s": round((end - start) * self.dt, 6),
                "mode": first["mode"],
                "held": first["held"],
                "start": self._endpoint(first),
                "end": self._endpoint(last),
            }
            print(
                "TRACE_SEG " + json.dumps(segment, separators=(",", ":")),
                flush=True,
            )
        final = self.samples[-1]
        summary = {
            "type": "result",
            "result": result,
            "final_frame": final["frame"],
            "max_lift_m": round(
                max(sample["lift_m"] for sample in self.samples), 6
            ),
            "ever_gripper_bowl_contact": any(
                sample["contact_count"] > 0 for sample in self.samples
            ),
            "final": self._endpoint(final),
        }
        print(
            "TRACE_RESULT " + json.dumps(summary, separators=(",", ":")),
            flush=True,
        )
        print("===== STIRFRY_TELEOP_TRACE_END =====\n", flush=True)
        print(
            "[수동 궤적 기록 완료] 위 BEGIN부터 END까지 전부 복사해 전달하세요.",
            flush=True,
        )
