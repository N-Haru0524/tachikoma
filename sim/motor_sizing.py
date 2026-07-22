"""
tachikoma / Phase 2b  ―  全12軸の必要スペック実測とモーター選定

戦略(段差越えシーケンスの成立性)は別課題。ここでは全12軸の必要スペックを実測し
型番に落とす。前提: 総重量1.3kg / ホイール径3cm(半径0.015) / 7.4V / 床µ=0.8。

軸:
  股ピッチ ×4 : 確定済み(Phase1: pull-up 1.7 N·m保守側=1脚集中, ×2=3.4)。記録引継ぎ。
  股ヨー   ×4 : 実測。据え切り(scrub)トルク / 操舵スルー(90°/0.3s)の所要トルク / 可動範囲。
  ホイール ×4 : 実測。必要トルク(平地加速 + 最大傾斜維持 + 敷居際=牽引限界) /
                必要回転数 / 連続 vs 瞬時ピーク。

計測の設計(モデル依存量・数値パラメータを実荷重から切り分ける):
  ・armature(関節の反映回転子慣性, 既定0.003)は「モーター自身の慣性」で選定入力とし
    ては循環参照。負荷トルクは M対角から armature を除いた機械慣性で評価する。
  ・ホイール damping(軸受抵抗の placeholder, 既定0.01)は 33 rad/s で 0.33 N·m にもなり
    実駆動負荷を覆い隠す。よって連続トルクは「傾斜での静的保持(車輪ブレーキ, ω=0)」で
    測り damping を無効化する。加速の機械分は F=ma から算出(軸受/ギヤ損失は効率と×2で吸収)。
  ・µ=0.8 はモデルの接線摩擦を書き換えず解析で適用(牽引限界 µ·N·r)。据え切りは接触の
    ねじれ摩擦で決まり接線µには依存しない(同軸操舵のため)。

使い方:
    uv run python sim/motor_sizing.py
"""
import numpy as np
import mujoco

from quad_model import make, WHEEL_R, STEP_X0
from quad_stability import leg_normal_force, stability

LEGS = ["RF", "LF", "RB", "LB"]

# ---- 入力パラメータ(確認済み) -------------------------------------------
MU = 0.8
V_CRUISE = 0.5               # 巡航速度 [m/s]
SLOPE_DEG = 10.0             # 車輪で維持する最大傾斜 [deg]
T_ACC = 0.5                  # 0→巡航 到達時間 [s] → a = V/T
STEER_ANGLE = np.pi / 2      # 操舵 90°
STEER_TIME = 0.3             # 90°を切る時間 [s]
MASS, G = 1.3, 9.81
WHEEL_CIRC = np.pi * (2 * WHEEL_R)
OMEGA_CRUISE = V_CRUISE / WHEEL_R        # 巡航角速度 [rad/s]
A_ACC = V_CRUISE / T_ACC                 # 加速度 [m/s²]

PITCH_PEAK = 1.7             # 股ピッチ pull-up 律速(Phase1 保守側, 1脚集中)[N·m]


# ---- 測定用モデル(摩擦は書き換えない=接触は安定。forcerange のみ拡張) --------
def make_meas(slope_deg=0.0):
    m, d, A, info = make(1.3)
    for n in LEGS:
        m.actuator(f"yawpos_{n}").forcerange[:] = [-10, 10]
        m.actuator(f"wheeldrv_{n}").forcerange[:] = [-5, 5]
    th = np.radians(slope_deg)
    m.opt.gravity[:] = [-G * np.sin(th), 0.0, -G * np.cos(th)]   # +x を上り勾配に
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d, A, info


def _hold(d, A, wheel=0.0):
    for n in LEGS:
        d.ctrl[A[f"yawpos_{n}"]] = 0.0
        d.ctrl[A[f"pitchpos_{n}"]] = 0.0
        d.ctrl[A[f"wheeldrv_{n}"]] = wheel


def mech_inertia(m, d, joint):
    """関節の機械慣性[kg·m²] = 関節空間質量行列の対角 − armature(placeholder除去)。"""
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    dof = m.joint(joint).dofadr[0]
    return float(M[dof, dof]) - float(m.dof_armature[dof])


# ============================================================================
# 股ヨー
# ============================================================================
def measure_yaw_scrub():
    """平地接地・車輪ブレーキで RF を準静的(3s)に 0→60°操舵し、車輪-床摩擦(接触の
    ねじれ抵抗)に抗する据え切りトルクを実測。荷重は操舵前(静定時)の値を採る。"""
    m, d, A, info = make_meas()
    _hold(d, A)
    for _ in range(800):
        mujoco.mj_step(m, d)
    load = leg_normal_force(m, d, "RF")
    peak = 0.0
    N = int(3.0 / m.opt.timestep)
    for k in range(N):
        d.ctrl[A["yawpos_RF"]] = min(1.0, (k + 1) / N) * (np.pi / 3)
        d.ctrl[A["wheeldrv_RF"]] = 0.0
        mujoco.mj_step(m, d)
        peak = max(peak, abs(d.actuator("yawpos_RF").force[0]))
    return {"scrub": peak, "load": load}


def yaw_slew_torque(scrub):
    """操舵スルー(90°/0.3s)の所要トルク = I_yaw_mech·α_peak + 据え切り。
    同軸操舵(車輪接地が操舵軸直下)＋細棒脚で機械慣性は極小 → 据え切り支配。"""
    m, d, A, info = make_meas()
    _hold(d, A)
    for _ in range(500):
        mujoco.mj_step(m, d)
    Iy = mech_inertia(m, d, "yaw_RF")
    alpha_peak = STEER_ANGLE * (np.pi / STEER_TIME) ** 2 / 2   # cosine 軌道の最大角加速度
    T_inertia = Iy * alpha_peak
    return {"I_yaw": Iy, "alpha": alpha_peak, "T_inertia": T_inertia, "T_slew": T_inertia + scrub}


# ============================================================================
# ホイール
# ============================================================================
def measure_wheel_continuous():
    """最大傾斜(10°)で車輪をブレーキ(ω=0)して静的保持し、重力に抗する保持トルクを
    実測(=登坂維持に要する連続トルク。ω=0 のため軸受damping の影響を受けない)。"""
    m, d, A, info = make_meas(slope_deg=SLOPE_DEG)
    _hold(d, A, wheel=0.0)
    for _ in range(2000):
        _hold(d, A, wheel=0.0)
        mujoco.mj_step(m, d)
    hold = np.mean([abs(d.actuator(f"wheeldrv_{n}").force[0]) for n in LEGS])
    return {"cont": hold, "v_drift": float(d.qvel[0]), "nsup": stability(m, d)["n_support"]}


def measure_traction_ceiling():
    """敷居際: 前輪を段差の壁に当てて前進駆動・停動させ、前輪荷重 N_front を実測 →
    牽引限界トルク µ·N_front·r を算出(モーターが出せるべき停動トルクの実用上限)。"""
    m, d, A, info = make_meas(slope_deg=0.0)
    _hold(d, A)
    for _ in range(500):
        mujoco.mj_step(m, d)
    for _ in range(5000):
        for n in LEGS:
            d.ctrl[A[f"wheeldrv_{n}"]] = 3.0
        mujoco.mj_step(m, d)
        ff = max(d.body(info["wheel_bid"][l]).xpos[0] for l in ("RF", "LF")) + WHEEL_R
        if ff >= STEP_X0 - 0.001:
            break
    for _ in range(int(1.0 / m.opt.timestep)):
        for n in LEGS:
            d.ctrl[A[f"wheeldrv_{n}"]] = 6.0
        mujoco.mj_step(m, d)
    n_front = np.mean([leg_normal_force(m, d, l) for l in ("RF", "LF")])
    return {"n_front": n_front, "ceiling": MU * n_front * WHEEL_R}


def confirm_cruise():
    """平地で駆動し、目標巡航速度に到達できることを確認(必要回転数の妥当性チェック)。"""
    m, d, A, info = make_meas(slope_deg=0.0)
    _hold(d, A)
    for _ in range(500):
        mujoco.mj_step(m, d)
    for _ in range(int(2.5 / m.opt.timestep)):
        for n in LEGS:
            d.ctrl[A[f"wheeldrv_{n}"]] = OMEGA_CRUISE
        mujoco.mj_step(m, d)
    return {"v": float(d.qvel[0])}


# ============================================================================
def rpm(w):
    return w * 60 / (2 * np.pi)


def main():
    ys = measure_yaw_scrub()
    yl = yaw_slew_torque(ys["scrub"])
    wc = measure_wheel_continuous()
    tr = measure_traction_ceiling()
    cr = confirm_cruise()

    N_leg = MASS * G / 4
    r_patch = 0.0015
    scrub_analytic = MU * N_leg * r_patch

    yaw_gov = max(ys["scrub"], yl["T_slew"])
    rpm_c = rpm(OMEGA_CRUISE)
    # ホイール 機械トルク(F=ma): 平地加速 と 傾斜加速(=連続+加速) を算出
    wheel_acc_flat = MASS * A_ACC / 4 * WHEEL_R
    wheel_peak_mech = MASS * (A_ACC + G * np.sin(np.radians(SLOPE_DEG))) / 4 * WHEEL_R

    L = "=" * 80
    print(L)
    print("Phase 2b: 全12軸 必要スペック実測 と モーター選定  (総重量1.3kg / 7.4V / µ=0.8)")
    print(L)
    print("\n■ 入力: 巡航 %.1f m/s / 最大傾斜 %.0f° / 0→巡航 %.1fs(a=%.2f m/s²) / 操舵 90°を%.1fs"
          % (V_CRUISE, SLOPE_DEG, T_ACC, A_ACC, STEER_TIME))
    print("  ホイール円周 π×3cm=%.4fm → 巡航 %.2f rad/s = %.0f rpm ／ 静立1脚荷重 N=%.2f N"
          % (WHEEL_CIRC, OMEGA_CRUISE, rpm_c, N_leg))

    print("\n── 股ヨー 実測 ──────────────────────────────────────────────────")
    print("  据え切りトルク(準静的操舵)             : %.4f N·m  (接地荷重 %.2f N)"
          % (ys["scrub"], ys["load"]))
    print("    ├ 接触ねじれ摩擦で決まる上側(接線µには非依存)。")
    print("    └ 解析 µ·N·r_patch(µ=%.1f, r=%.1fmm)=%.4f N·m を下側目安として併記。"
          % (MU, r_patch * 1000, scrub_analytic))
    print("  操舵スルー 90°/0.3s の所要トルク       : %.4f N·m" % yl["T_slew"])
    print("    ├ 機械慣性 I_yaw=%.2e kg·m²(同軸操舵＋細棒脚で極小) × α=%.0f rad/s² = %.4f N·m"
          % (yl["I_yaw"], yl["alpha"], yl["T_inertia"]))
    print("    └ ∴ スルーの慣性分は無視でき、据え切りが支配。")
    print("  → 股ヨー律速 = %.4f N·m ／ 可動範囲 = ±90°(swerve クラブ+θ回転を包含)" % yaw_gov)

    print("\n── ホイール 実測 ───────────────────────────────────────────────")
    print("  傾斜%.0f° 登坂維持(連続, 車輪ブレーキ静保持で実測)  : %.4f N·m/輪 (接地%d脚, 滑落%.3f m/s)"
          % (SLOPE_DEG, wc["cont"], wc["nsup"], wc["v_drift"]))
    print("  平地 加速 0→巡航 (機械分, F=ma)                    : %.4f N·m/輪" % wheel_acc_flat)
    print("  傾斜%.0f° 加速 (瞬時ピーク=連続+加速, 機械分)        : %.4f N·m/輪" % (SLOPE_DEG, wheel_peak_mech))
    print("  敷居際 牽引限界 µ·N_front·r (停動の実用上限, 実測N) : %.4f N·m/輪 (前輪荷重 %.2f N)"
          % (tr["ceiling"], tr["n_front"]))
    print("  巡航到達確認(平地駆動)                            : %.2f m/s (目標 %.1f)"
          % (cr["v"], V_CRUISE))
    print("  → 連続=%.4f / 瞬時ピーク(機械)=%.4f / 牽引上限=%.4f N·m、必要 %.0f rpm"
          % (wc["cont"], wheel_peak_mech, tr["ceiling"], rpm_c))

    _print_table(ys, yl, yaw_gov, wc, wheel_peak_mech, tr, rpm_c, N_leg)


def _print_table(ys, yl, yaw_gov, wc, wheel_peak, tr, rpm_c, N_leg):
    print("\n" + "=" * 80)
    print("【12軸 サイジング一覧】(必要値 と 安全率×2)")
    print("=" * 80)
    print("● 股ピッチ ×4")
    print("    必要トルク : ピーク %.2f N·m(段差 pull-up=1脚集中, Phase1保守側) / 連続:車体支持"
          % PITCH_PEAK)
    print("    速度/回転  : pull-up 低速     可動範囲: ±80°    分担荷重: 最悪=全1.3kg集中")
    print("    サイジング : ×2 = %.2f N·m" % (PITCH_PEAK * 2))
    print("● 股ヨー   ×4")
    print("    必要トルク : 据え切り %.3f / 操舵スルー %.3f(慣性分ほぼ0) N·m"
          % (ys["scrub"], yl["T_slew"]))
    print("    速度/回転  : 90°/0.3s        可動範囲: ±90°    分担荷重: %.2f N(操舵は持上げ無)"
          % N_leg)
    print("    サイジング : ×2 = %.3f N·m" % (yaw_gov * 2))
    print("● ホイール ×4")
    print("    必要トルク : 連続 %.4f / 機械ピーク %.4f / 牽引上限 %.4f N·m/輪"
          % (wc["cont"], wheel_peak, tr["ceiling"]))
    print("    速度/回転  : %.0f rpm(巡航)   可動範囲: 連続回転  分担荷重: %.2f N(1輪)"
          % (rpm_c, N_leg))
    print("    サイジング : 連続×2=%.4f / ストールは牽引上限 %.3f まで確保が目標"
          % (wc["cont"] * 2, tr["ceiling"]))

    print("\n" + "=" * 80)
    print("【型番提案】(7.4V 2セル駆動前提, Feetech バス混載)")
    print("=" * 80)
    print("  股ピッチ ×4 : Feetech STS3250 級 (ストール 3〜4 N·m)")
    print("      必要 ×2=%.2f N·m。段差 pull-up が全12軸の律速。3250 射程内。§8 の確定を維持。"
          % (PITCH_PEAK * 2))
    print("  股ヨー   ×4 : Feetech STS3215 級 (ストール 1.5〜3 N·m)")
    print("      必要 ×2=%.3f N·m と極小(操舵は車体を持上げず、同軸設計で据え切りも小)。" % (yaw_gov * 2))
    print("      トルクは大幅に余るが、STS3250 とバス・寸法互換で配線一系統にできる 3215 を選択。")
    print("      位置サーボで操舵精度・保持剛性も確保。可動範囲 ±90°。")
    print("  ホイール ×4 : ギアードDCモーター＋エンコーダ or 連続回転サーボ")
    print("      必要 %.0f rpm、トルクは連続 %.4f・機械ピーク %.4f・牽引上限 %.3f N·m/輪 と極小"
          % (rpm_c, wc["cont"], wheel_peak, tr["ceiling"]))
    print("      (径3cmで低トルク高回転側)。律速は回転数と効率。目安: 無負荷 ≳%.0f rpm、"
          % (rpm_c * 1.3))
    print("      ストール ≳%.3f N·m(牽引限界を使い切れる)。例: Pololu 等の小型ギヤードDC"
          % tr["ceiling"])
    print("      ＋磁気エンコーダ、または FS90R 級連続回転サーボ。高トルクサーボは過剰。")
    print("\n  ・混載でも Feetech STS 同系はバス互換(TTL/RS485 デイジーチェーン一系統)。§8と整合。")
    print("  ・7.4V でストールトルク確保。股ピッチのみが律速、他10軸は大きく余裕。")
    print("  ・重量: 股ピッチを3250にすると +~100g/4軸。総重量~1.3kg 内で pull-up~1.7 N·m と収束。")


if __name__ == "__main__":
    main()
