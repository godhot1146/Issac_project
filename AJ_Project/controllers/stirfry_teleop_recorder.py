"""볶음 그릇 수동 파지 동작을 자동 시퀀스 분석용으로 기록한다.

키 입력 변화 시점과 5 Hz 주기 상태만 메모리에 보존하고, 출력할 때 같은
키/접촉 상태가 이어지는 샘플을 구간으로 압축한다. 출력된 TRACE 블록만
복사해도 TSC 목표, 실제 관절/그리퍼 자세, 그릇 자세, 접촉 및 상승 시점을
재구성할 수 있다.
"""
import json

import numpy as np
from isaacgym import gymapi
from scipy.spatial.transform import Rotation


class StirfryTeleopRecorder:
    FORMAT_VERSION = 2
    SAMPLE_RATE_HZ = 5.0
    MAX_SEGMENT_SAMPLES = 30
    LIFT_THRESHOLD_M = 0.015
    MAX_OUTPUT_PART_CHARS = 10000
    # PART 머리말/꼬리말과 줄바꿈을 위한 보수적인 여유 공간.
    OUTPUT_PART_OVERHEAD_CHARS = 256

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
        self.frame = 0
        self.sample_interval_frames = max(
            1, int(round(1.0 / (self.dt * self.SAMPLE_RATE_HZ)))
        )
        self._last_input_signature = None

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

        bowl_state = self._body_pose(bowl_actor, 0)
        self.initial_bowl_z = bowl_state["p"][2]
        print(
            f"[수동 궤적 기록 시작] 저부하 {self.SAMPLE_RATE_HZ:.0f}Hz 기록을 "
            "시작합니다. 키 입력 변화는 즉시 기록합니다. "
            "성공 후 ENTER를 누르면 복사용 TRACE 블록이 출력됩니다.",
            flush=True,
        )
        self.record(force=True)

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

    def _body_pose(self, actor, body_index):
        states = self.gym.get_actor_rigid_body_states(
            self.env, actor, gymapi.STATE_POS
        )
        state = states[body_index]
        return {
            "p": self._vec3(state["pose"]["p"]),
            "q": self._quat(state["pose"]["r"]),
        }

    def _contact_state(self):
        count = 0
        for contact in self.gym.get_env_rigid_contacts(self.env):
            body0 = int(contact["body0"])
            body1 = int(contact["body1"])
            if {body0, body1} != {self.gripper_env_body, self.bowl_env_body}:
                continue
            count += 1
        return count

    def _commanded_joints(self, actual_joints):
        if self.teleop.mode == "jsc":
            return np.asarray(self.teleop.joint_target, dtype=np.float64)
        if self.teleop.mode == "tsc" and self.arm._last_q is not None:
            return np.asarray(self.arm._last_q, dtype=np.float64)
        return np.asarray(actual_joints, dtype=np.float64)

    def record(self, force=False):
        """키 상태 변화 또는 5 Hz 주기에만 물리 상태를 읽어 기록한다."""
        if not self.active:
            return

        input_signature = (self.teleop.mode, tuple(sorted(self.teleop.held)))
        input_changed = input_signature != self._last_input_signature
        periodic = self.frame % self.sample_interval_frames == 0
        should_sample = bool(force or input_changed or periodic)
        self._last_input_signature = input_signature
        current_frame = self.frame
        self.frame += 1
        if not should_sample:
            return

        gripper = self._body_pose(self.arm.actor, self.gripper_body_index)
        bowl = self._body_pose(self.bowl_actor, 0)
        contact_count = self._contact_state()
        actual_joints = self.arm.current_joints()
        target_q = Rotation.from_matrix(self.teleop.ori_R).as_quat()
        lift_m = float(bowl["p"][2] - self.initial_bowl_z)

        self.samples.append(
            {
                "frame": current_frame,
                "mode": self.teleop.mode,
                "held": sorted(self.teleop.held),
                "target_p_base": self._rounded(self.teleop.cart_target),
                "target_q_base_xyzw": self._rounded(target_q),
                "q_cmd_rad": self._rounded(
                    self._commanded_joints(actual_joints)
                ),
                "q_actual_rad": self._rounded(actual_joints),
                "gripper_p_world": self._rounded(gripper["p"]),
                "gripper_q_world_xyzw": self._rounded(gripper["q"]),
                "bowl_p_world": self._rounded(bowl["p"]),
                "bowl_q_world_xyzw": self._rounded(bowl["q"]),
                "contact_count": contact_count,
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
            "gripper_p_world",
            "gripper_q_world_xyzw",
            "bowl_p_world",
            "bowl_q_world_xyzw",
            "contact_count",
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
            span_full = index - start >= self.MAX_SEGMENT_SAMPLES
            if signature_changed or span_full:
                end = index - 1
                segments.append((start, end))
                start = index
        if not segments or segments[-1][1] != len(self.samples) - 1:
            segments.append((start, len(self.samples) - 1))
        return segments

    @classmethod
    def _output_parts(cls, lines):
        """TRACE 줄을 10,000자 이하의 복사 가능한 파트로 묶는다."""
        payload_limit = (
            cls.MAX_OUTPUT_PART_CHARS - cls.OUTPUT_PART_OVERHEAD_CHARS
        )
        parts = []
        current = []
        current_chars = 0
        for line in lines:
            line_chars = len(line) + 1  # 출력 줄바꿈 포함
            if line_chars > payload_limit:
                raise RuntimeError(
                    "단일 TRACE 줄이 출력 파트 제한을 초과했습니다: "
                    f"{line_chars} chars"
                )
            if current and current_chars + line_chars > payload_limit:
                parts.append(current)
                current = []
                current_chars = 0
            current.append(line)
            current_chars += line_chars
        if current:
            parts.append(current)
        return parts

    @classmethod
    def _print_output_parts(cls, lines):
        parts = cls._output_parts(lines)
        total = len(parts)
        print(
            f"\n[수동 궤적 기록] 총 {total}개 파트로 나눠 출력합니다. "
            "각 PART의 BEGIN부터 END까지 한 파트씩 복사하세요.",
            flush=True,
        )
        for index, part_lines in enumerate(parts, start=1):
            begin = (
                "===== STIRFRY_TELEOP_TRACE_PART "
                f"{index}/{total} BEGIN ====="
            )
            end = (
                "===== STIRFRY_TELEOP_TRACE_PART "
                f"{index}/{total} END ====="
            )
            block = "\n".join([begin, *part_lines, end])
            # print가 덧붙이는 마지막 줄바꿈까지 10,000자 제한에 포함한다.
            if len(block) + 1 > cls.MAX_OUTPUT_PART_CHARS:
                raise RuntimeError(
                    f"TRACE PART {index}/{total}가 출력 제한을 초과했습니다: "
                    f"{len(block) + 1} chars"
                )
            print(block, flush=True)
            print(
                f"[PART {index}/{total} 끝 — 위 블록을 복사하세요]",
                flush=True,
            )

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
            "frames": self.frame,
            "samples": len(self.samples),
            "sample_rate_hz": self.SAMPLE_RATE_HZ,
            "duration_s": round(max(0, self.frame - 1) * self.dt, 6),
            "cart_step_m_per_frame": float(self.teleop.cart_step),
            "ori_step_deg_per_frame": round(
                float(np.rad2deg(self.teleop.ori_step)), 6
            ),
            "joint_step_rad_per_frame": float(self.teleop.joint_step),
            "initial_bowl_z_world": round(self.initial_bowl_z, 6),
            "lift_threshold_m": self.LIFT_THRESHOLD_M,
            "quaternion_order": "xyzw",
        }
        output_lines = [
            "TRACE_META " + json.dumps(meta, separators=(",", ":")),
            "TRACE_START "
            + json.dumps(
                self._endpoint(self.samples[0]), separators=(",", ":")
            ),
        ]
        for sequence, (start, end) in enumerate(self._segments()):
            first = self.samples[start]
            last = self.samples[end]
            segment = {
                "type": "segment",
                "seq": sequence,
                "f0": first["frame"],
                "f1": last["frame"],
                "duration_s": round(
                    (last["frame"] - first["frame"]) * self.dt, 6
                ),
                "mode": first["mode"],
                "held": first["held"],
                "start": self._endpoint(first),
                "end": self._endpoint(last),
            }
            output_lines.append(
                "TRACE_SEG " + json.dumps(segment, separators=(",", ":"))
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
        output_lines.append(
            "TRACE_RESULT " + json.dumps(summary, separators=(",", ":"))
        )
        self._print_output_parts(output_lines)
        print(
            "[수동 궤적 기록 완료] 모든 PART를 번호 순서대로 전달하세요.",
            flush=True,
        )
