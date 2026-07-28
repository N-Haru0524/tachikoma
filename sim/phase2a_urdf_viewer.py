"""
tachikoma / Phase 2a  案(b) 動的リフト  ビューア(URDF直読みモデル)

URDF直読み4脚(立位)で、案(b)の単隅リフト挙動を可視化する。
成立域の推奨(中庸開脚・遅リフト)と、不可域(速リフト/COM高)を見比べられる。

使い方:
    uv run python sim/phase2a_urdf_viewer.py            # 推奨(成立)を巡回
    uv run python sim/phase2a_urdf_viewer.py fast       # 速リフト(不可域)
    uv run python sim/phase2a_urdf_viewer.py headless    # 画面なし数値検証

局面: STAND(4輪) → LIFT(前脚上げ→対角抜け) → HOLD(段へ運ぶ想定) → BACK(戻して回復)。
"""
import sys
import time
import numpy as np
import mujoco

import urdf_model as U
from phase2a_urdf import _lead_leg, _lift_pitch, _tilt
from quad_stability import leg_normal_force

PRESETS = {
    "recommended": dict(splay=0.45, P=0.6, lift_time=0.30),
    "fast": dict(splay=0.20, P=0.4, lift_time=0.15),
}
SEG = 2.0


def _setup(preset):
    p = PRESETS[preset]
    m, d, A = U.make_stand(P=p["P"], yaw_splay=p["splay"], settle=1200)
    stance = {f: U.PITCH_SIGN[f] * p["P"] for f in U.FEET}
    ysign = {"feet": +1, "feet_2": +1, "feet_1": -1, "feet_3": -1}
    yaw_ctrl = {f: ysign[f] * p["splay"] for f in U.FEET}
    lead = _lead_leg(m, d)
    lift_target, _ = _lift_pitch(m, d, lead, stance)
    return m, d, A, stance, yaw_ctrl, lead, lift_target, p["lift_time"]


def _phase_ctrl(t, stance, lead, lift_target, lift_time):
    ph = t % (4 * SEG)
    if ph < SEG:
        return stance[lead], "STAND  4脚で立つ"
    elif ph < 2 * SEG:
        a = min(1.0, (ph - SEG) / lift_time)
        return stance[lead] + a * (lift_target - stance[lead]), "LIFT   前脚上げ→対角抜け(接地2輪)"
    elif ph < 3 * SEG:
        return lift_target, "HOLD   滞空(段へ足を運ぶ想定)"
    else:
        a = min(1.0, (ph - 3 * SEG) / lift_time)
        return lift_target + a * (stance[lead] - lift_target), "BACK   足を戻す→回復"


def run_headless(preset="recommended"):
    m, d, A, stance, yaw_ctrl, lead, lift_target, lt = _setup(preset)
    dt = m.opt.timestep
    last = None
    peak = 0.0
    for k in range(int(4 * SEG / dt)):
        t = k * dt
        p, tag = _phase_ctrl(t, stance, lead, lift_target, lt)
        for f in U.FEET:
            d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
            d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = yaw_ctrl[f]
        d.ctrl[A[f"pitchpos_{lead}"]] = p
        mujoco.mj_step(m, d)
        peak = max(peak, _tilt(m, d))
        if tag != last:
            nsup = sum(1 for f in U.FEET if leg_normal_force(m, d, f) > 0.05)
            print("  [%4.1fs] %-34s tilt=%4.1f° 接地=%d bz=%.3f" %
                  (t, tag, _tilt(m, d), nsup, d.body("body").xpos[2]))
            last = tag
    print("  preset=%s → ピーク傾き=%.1f° 最終=%.1f°" % (preset, peak, _tilt(m, d)))


def main(preset="recommended"):
    m, d, A, stance, yaw_ctrl, lead, lift_target, lt = _setup(preset)
    print(__doc__); print("preset =", preset, PRESETS[preset], " lead =", lead)
    import mujoco.viewer
    last = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            p, tag = _phase_ctrl(t, stance, lead, lift_target, lt)
            for f in U.FEET:
                d.ctrl[A[f"pitchpos_{f}"]] = stance[f]
                d.ctrl[A[f"yawpos_{U.YAW_OF[f]}"]] = yaw_ctrl[f]
            d.ctrl[A[f"pitchpos_{lead}"]] = p
            if tag != last:
                print("  [%5.1fs] %s" % (t, tag)); last = tag
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(dt if (dt := m.opt.timestep) else 0.002)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "recommended"
    if arg == "headless":
        run_headless("recommended"); print(); run_headless("fast")
    else:
        main(arg if arg in PRESETS else "recommended")
