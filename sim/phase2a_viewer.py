"""
tachikoma / Phase 2a  案(b) 動的リフト段差越え ビューア

CAD由来4脚モデル(quad_cad, メッシュ視覚形状つき)で、案b の単隅リフト挙動を可視化する。
成立域の推奨動作点(広トレッド・中庸リフト)と、不可域(狭トレッド)を見比べられる。

使い方 (uv):
    uv run python sim/phase2a_viewer.py               # 推奨動作点(成立)を巡回表示
    uv run python sim/phase2a_viewer.py narrow        # 狭トレッド(不可域)= 傾き過大を可視化
    uv run python sim/phase2a_viewer.py headless      # 画面なしで挙動を数値検証

巡回する局面:
  [STAND] 4脚で立つ(全輪接地)。
  [LIFT ] 前右(RF)を短時間で振り上げ = 対角(LB)が抜け、LF-RB の対角線バランスへ。
  [HOLD ] 段上へ足を運ぶ想定の滞空。車体は RF 隅へ有限角だけ傾く(倒れきらない)。
  [BACK ] 足を戻す = 4輪接地へ回復し水平へ。
"""
import sys
import time
import numpy as np
import mujoco

from quad_cad import make, ELL
from phase2a_liftover import (tilt_deg, _settle_stand, LIFT_LEG, LEGS, LIFT_CLEAR_Z)
from quad_stability import leg_normal_force

# (tread, stance_pitch, lift_time) プリセット
PRESETS = {
    "recommended": (0.18, -0.50, 0.25),   # 成立域の実用推奨
    "narrow": (0.10, -0.30, 0.40),        # 不可域(狭トレッド+遅リフト)= 大きく傾く/転倒側
}

SEG = 2.2   # 各局面の秒数


def _lift_pitch(m, d, sp):
    hip_z = d.body(f"hip_{LIFT_LEG}").xpos[2]
    c = np.clip((hip_z - LIFT_CLEAR_Z) / ELL, -1.0, 1.0)
    return max(-2.79, -float(np.arccos(c)))


def cycle(preset="recommended"):
    tread, sp, lift_time = PRESETS[preset]
    m, d, A, info = make(tread=tread, stance_pitch=sp)
    _settle_stand(m, d, A, info, sp)
    lift_p = _lift_pitch(m, d, sp)
    return m, d, A, info, sp, lift_p, lift_time


def _target_pitch(t, sp, lift_p, lift_time):
    """周期 t[s] に応じた RF 目標ピッチと局面タグ。"""
    phase = t % (4 * SEG)
    if phase < SEG:
        return sp, "STAND  4脚で立つ(全輪接地)"
    elif phase < 2 * SEG:
        a = min(1.0, (phase - SEG) / lift_time)
        return sp + a * (lift_p - sp), "LIFT   RF振り上げ→対角LB抜け→LF-RB対角線バランス"
    elif phase < 3 * SEG:
        return lift_p, "HOLD   滞空(車体はRF隅へ有限角だけ傾く=倒れきらない)"
    else:
        a = min(1.0, (phase - 3 * SEG) / lift_time)
        return lift_p + a * (sp - lift_p), "BACK   足を戻す→4輪接地へ回復・水平へ"


def run_headless(preset="recommended", cycles=1.0):
    m, d, A, info, sp, lift_p, lift_time = cycle(preset)
    dt = m.opt.timestep
    tread = PRESETS[preset][0]
    print("preset=%s  tread=%.0fmm stance=%.2f lift_time=%.2fs" % (preset, tread * 1000, sp, lift_time))
    last = None
    peak_tilt = 0.0
    n = int(cycles * 4 * SEG / dt)
    for k in range(n):
        t = k * dt
        p, tag = _target_pitch(t, sp, lift_p, lift_time)
        d.ctrl[A[f"pitchpos_{LIFT_LEG}"]] = p
        for nleg in LEGS:
            if nleg != LIFT_LEG:
                d.ctrl[A[f"pitchpos_{nleg}"]] = sp
            d.ctrl[A[f"wheeldrv_{nleg}"]] = 0.0
        mujoco.mj_step(m, d)
        peak_tilt = max(peak_tilt, tilt_deg(d))
        if tag != last:
            nsup = sum(1 for nl in LEGS if leg_normal_force(m, d, nl) > 0.05)
            print("  [%5.2fs] %-52s tilt=%4.1f° 接地輪=%d bz=%.3f" %
                  (t, tag, tilt_deg(d), nsup, d.body("body").xpos[2]))
            last = tag
    print("  → 1周ピーク傾き = %.1f°  (final %.1f°)" % (peak_tilt, tilt_deg(d)))
    print("headless OK")


def main(preset="recommended"):
    m, d, A, info, sp, lift_p, lift_time = cycle(preset)
    print(__doc__)
    print("preset =", preset, " (tread=%.0fmm, stance=%.2f, lift=%.2fs)" %
          (PRESETS[preset][0] * 1000, sp, lift_time))
    import mujoco.viewer
    last = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            p, tag = _target_pitch(t, sp, lift_p, lift_time)
            d.ctrl[A[f"pitchpos_{LIFT_LEG}"]] = p
            for nleg in LEGS:
                if nleg != LIFT_LEG:
                    d.ctrl[A[f"pitchpos_{nleg}"]] = sp
                d.ctrl[A[f"wheeldrv_{nleg}"]] = 0.0
            if tag != last:
                print("  [%5.1fs] %s" % (t, tag))
                last = tag
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "recommended"
    if arg == "headless":
        run_headless("recommended")
        print()
        run_headless("narrow")
    else:
        main(arg if arg in PRESETS else "recommended")
