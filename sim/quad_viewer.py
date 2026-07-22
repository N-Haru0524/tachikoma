"""
tachikoma / Phase 2  4脚フルモデルのビューア

使い方 (uv):
    uv run python sim/quad_viewer.py            # デモ(挙動を巡回表示)
    uv run python sim/quad_viewer.py stand      # 4脚で立って静定するだけ
    uv run python sim/quad_viewer.py headless    # 画面なしで制御ロジックだけ検証

デモが巡回する挙動(要件 §3,§4,§6 / Phase 2 の検証内容を可視化):
  [STAND]  4脚で立つ。全輪接地・重心中央(支持多角形の中に COM)。
  [SHIFT]  股ピッチを揃えて前後に振る = 前後方向の荷重移動(pull-up 前の荷重移し)。
  [LIFT ]  1脚を上げようとする = 対角シーソーで車体が傾く。左右の重心移動ができない
           ため単隅の脚上げは静的に不安定、という Phase 2 の知見を可視化。
  [STEP ]  straddle: 前輪1つを 3cm 段の上面に置き、その脚を段上でアンカーして
           股ピッチで車体を引き上げる(pull-up)= 段越えの中核動作。
  [CRAB ]  全輪を +90°操舵して横move = swerve のクラブ(向けて走る)。§3 の移動系。
"""
import sys
import time
import numpy as np
import mujoco

from quad_model import make, STAND_BODY_Z
from quad_sequence import (Ctrl, LEGS, STRADDLE_PITCH, PULLUP_TARGET,
                           make_straddle, _rpy_deg)
from quad_stability import stability, leg_normal_force

CYCLE = 30.0
SEG = CYCLE / 5.0     # 各挙動 6 秒


def _reset_stand(m, d):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def controller(m, d, A, info, ctrl, t, mode):
    """t[s] に応じて ctrl(各脚 yaw/pitch/wheel 目標)を設定し、タグを返す。"""
    if mode == "stand":
        for n in LEGS:
            ctrl.pitch[n] = 0.0
            ctrl.wheel[n] = 0.0
        return "STAND  4脚で立つ"

    phase = t % CYCLE
    seg = int(phase // SEG)
    s = phase - seg * SEG

    # 各セグメント開始時の初期化はメインループ側(reset)で処理
    if seg == 0:                                   # STAND
        for n in LEGS:
            ctrl.yaw[n] = 0.0
            ctrl.pitch[n] = 0.0
            ctrl.wheel[n] = 0.0
        return "STAND  4脚で立つ(全輪接地・COM中央)"
    elif seg == 1:                                 # SHIFT (前後荷重移動)
        a = 0.35 * np.sin(2 * np.pi * s / SEG)
        for n in LEGS:
            ctrl.pitch[n] = a
            ctrl.wheel[n] = 0.0
        return "SHIFT  前後の荷重移動(股ピッチ)"
    elif seg == 2:                                 # LIFT (単隅上げ→傾く)
        a = min(1.0, s / 2.0)
        ctrl.pitch["RF"] = a * (-0.9)
        return "LIFT   1脚上げ→対角シーソーで傾く(左右重心移動できず)"
    elif seg == 3:                                 # STEP (straddle pull-up)
        # このセグメントは reset で straddle 姿勢に入れてある。RF を引き上げる。
        a = min(1.0, s / 3.5)
        ctrl.pitch["RF"] = STRADDLE_PITCH + a * (PULLUP_TARGET - STRADDLE_PITCH)
        ctrl.wheel["RF"] = 0.0
        return "STEP   段上アンカー→pull-upで車体を段へ引き上げ"
    else:                                          # CRAB (swerve 横move)
        for n in LEGS:
            ctrl.yaw[n] = 1.4
        drive = 6.0 if s > 1.0 else 0.0            # 操舵が入ってから駆動
        for n in LEGS:
            ctrl.wheel[n] = drive
        return "CRAB   swerve 横move(全輪+90°操舵して転がる)"


def _enter_segment(m, d, A, info, ctrl, seg):
    """セグメント切替時の姿勢初期化。"""
    if seg == 3:
        # straddle 姿勢(RF を段上に置いた状態)へ
        sim2 = make_straddle("RF")
        m2, d2, A2, info2, ctrl2 = sim2
        d.qpos[:] = d2.qpos
        d.qvel[:] = 0
        for n in LEGS:
            ctrl.yaw[n] = 0.0
            ctrl.pitch[n] = STRADDLE_PITCH if n == "RF" else 0.0
            ctrl.wheel[n] = 0.0
        mujoco.mj_forward(m, d)
    else:
        _reset_stand(m, d)
        for n in LEGS:
            ctrl.yaw[n] = 0.0
            ctrl.pitch[n] = 0.0
            ctrl.wheel[n] = 0.0


def run_headless(m, d, A, info, ctrl, seconds=CYCLE):
    """画面なしで1周ぶん回し、制御ロジックとモデルが破綻しないことを検証。"""
    dt = m.opt.timestep
    last_seg = -1
    t = 0.0
    n = int(seconds / dt)
    for k in range(n):
        seg = int((t % CYCLE) // SEG)
        if seg != last_seg:
            _enter_segment(m, d, A, info, ctrl, seg)
            last_seg = seg
        tag = controller(m, d, A, info, ctrl, t, "demo")
        ctrl.apply()
        mujoco.mj_step(m, d)
        if k % int(SEG / dt) == 0:
            s = stability(m, d)
            roll, pitch = _rpy_deg(d.qpos[3:7])
            print("[%4.1fs] %-46s bz=%.3f roll=%+5.1f pitch=%+5.1f nsup=%d margin=%s"
                  % (t, tag, d.body("body").xpos[2], roll, pitch, s["n_support"],
                     ("%+.3f" % s["margin"]) if s["margin"] is not None else "None"))
        t += dt
    print("headless OK (%d steps)" % n)


def main(mode="demo"):
    m, d, A, info = make(1.3)
    _reset_stand(m, d)
    ctrl = Ctrl(m, d, A, info)
    print(__doc__)
    print("mode =", mode)

    if mode == "headless":
        run_headless(m, d, A, info, ctrl)
        return

    last_seg = -1
    last_tag = None
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            if mode == "stand":
                tag = controller(m, d, A, info, ctrl, t, "stand")
            else:
                seg = int((t % CYCLE) // SEG)
                if seg != last_seg:
                    _enter_segment(m, d, A, info, ctrl, seg)
                    last_seg = seg
                tag = controller(m, d, A, info, ctrl, t, "demo")
            if tag != last_tag:
                print("  [%5.1fs] %s" % (t, tag))
                last_tag = tag
            ctrl.apply()
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode == "demo":
        import mujoco.viewer  # noqa (画面モードのみ必要)
    main(mode)
