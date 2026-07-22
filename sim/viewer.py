"""
tachikoma / Phase 1  脚1本モデル (v2: 膝なし棒脚 + 足首受動サス) のビューア

使い方 (uv):
    uv run python sim/viewer.py            # デモ (股ヨー/股ピッチ/段差pull-upを巡回)
    uv run python sim/viewer.py passive    # 放置 (重力で落ち着く様子だけ)

デモが巡回する挙動:
    [YAW ]  股ヨーを左右にスイープ  = 「脚を向ける」操舵 (剛サーボ)
    [PITCH] 股ピッチを前後に振る     = 車高/姿勢, お辞儀・伸びの仕草 (主軸)
    [STEP]  車輪を3cm段の上面に置いた状態から股ピッチで車体を引き上げる (pull-up)
            = 膝なしで段を越える動作。足首サスが接地衝撃を吸収。
"""
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from torque_probe import PULLUP_INIT, PULLUP_TARGET, PULLUP_RAMP_S

XML = "sim/leg_single.xml"


def main(mode="demo"):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    A = {m.actuator(i).name: i for i in range(m.nu)}
    qsx = m.joint("slide_x").qposadr[0]
    qsz = m.joint("slide_z").qposadr[0]
    qpi = m.joint("hip_pitch").qposadr[0]

    CYCLE = 21.0  # 7秒 × 3挙動

    def enter_step():
        # 車輪を段の縁上に置いた初期姿勢へ
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[qsx] = PULLUP_INIT["slide_x"]
        d.qpos[qsz] = PULLUP_INIT["slide_z"]
        d.qpos[qpi] = PULLUP_INIT["pitch"]
        mujoco.mj_forward(m, d)

    def controller(t):
        if mode != "demo":
            return "PASSIVE"
        phase = t % CYCLE
        if phase < 7.0:                       # [YAW]
            d.ctrl[A["hip_yaw_pos"]] = 1.2 * np.sin(2 * np.pi * phase / 7.0)
            d.ctrl[A["hip_pitch_pos"]] = 0.30
            d.ctrl[A["wheel_drive"]] = 0.0
            return "YAW   脚を向ける"
        elif phase < 14.0:                    # [PITCH]
            d.ctrl[A["hip_yaw_pos"]] = 0.0
            d.ctrl[A["hip_pitch_pos"]] = 0.30 + 0.45 * np.sin(2 * np.pi * (phase - 7.0) / 7.0)
            d.ctrl[A["wheel_drive"]] = 0.0
            return "PITCH 前後に振る/お辞儀"
        else:                                 # [STEP] pull-up
            s = phase - 14.0
            cmd = min(PULLUP_INIT["pitch"] + (PULLUP_TARGET - PULLUP_INIT["pitch"]) * (s / PULLUP_RAMP_S),
                      PULLUP_TARGET)
            d.ctrl[A["hip_yaw_pos"]] = 0.0
            d.ctrl[A["hip_pitch_pos"]] = cmd
            d.ctrl[A["wheel_drive"]] = 0.0
            return "STEP  3cm段差 pull-up"

    print(__doc__)
    print("mode = %s" % mode)
    last_tag = None
    cycle_idx = -1
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            phase = t % CYCLE
            ci = int(t // CYCLE)
            # サイクル開始で初期化、STEP フェーズ開始で pull-up 姿勢へ
            if ci != cycle_idx:
                cycle_idx = ci
                mujoco.mj_resetDataKeyframe(m, d, 0)
                last_tag = "_reset"
            if mode == "demo" and phase >= 14.0 and last_tag != "STEP  3cm段差 pull-up":
                enter_step()
            tag = controller(t)
            if tag != last_tag:
                pk = abs(d.actuator("hip_pitch_pos").force[0])
                print("  [%5.1fs] %s   (股ピッチtau=%.2f N*m)" % (t, tag, pk))
                last_tag = tag
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    main(mode)
