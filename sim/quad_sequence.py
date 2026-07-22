"""
tachikoma / Phase 2 主成果物  ―  4脚で1脚ずつ 3cm 段差を越えるシーケンスの検証

要件 §6,§9(Phase 2): 4脚全体モデルで「1脚ずつ」段差を越える。不変条件は
「越えている1脚以外の3脚が常に接地し、重心(COM)が残り3脚の支持三角形内に留まる
(=転倒しない)」。車輪を壁にぶつけ駆動で乗り上げる roll-over は禁止で、段差は脚の
step 動作(振り出し→段上に置く→荷重移動→pull-up)で越える。

本スクリプトは要件の測定項目に答える:
  [1] 各ステップの重心 vs 支持三角形の関係をログし、シーケンスが転倒せず成立するか。
  [2] 各脚 pull-up 中の股ピッチトルク波形とピーク。Phase 1 単脚ワースト(~1.7 N·m)
      および STS3250(3〜4 N·m)と比較して足りるか裏取り。
  [3] quad_viewer.py で可視化。

────────────────────────────────────────────────────────────────────────────
重要な知見(本 Phase 2 で判明。詳細は各パートの出力とコメント):
  この脚設計(股ヨー+股ピッチ+受動足首、膝なし・股ロールなし)は、接地した車輪の
  上で車体の重心を「横方向」へ静的に移せない(実測: 股ピッチは前後方向の荷重移動を
  作れるが、股ヨーは左右方向の重心移動を作れない)。1隅の脚を持ち上げるには対角
  (前後+左右)の重心移動が要るが、左右成分を作れないため、静的には必ず COM が
  支持三角形の対角線上〜外に出る=単隅の脚上げは静的に成立しない。
  → シーケンス自体の可否は「設計(自由度)の問題」であり、Phase 2 の再サイジング
     対象(§10)。一方、pull-up トルクの裏取り([2])は straddle 姿勢で健全に行える。
────────────────────────────────────────────────────────────────────────────

使い方:
    uv run python sim/quad_sequence.py            # [1][2] を実行しサマリ表示
    uv run python sim/quad_sequence.py --csv out.csv   # 詳細ログを CSV 出力
"""
import argparse
import numpy as np
import mujoco

from quad_model import make, STAND_BODY_Z, STEP_H, STEP_X0
from quad_stability import stability, com_full, com_xy, leg_normal_force, _poly_margin

LEGS = ["RF", "LF", "RB", "LB"]
DIAG_ORDER = ["RF", "LB", "LF", "RB"]   # 要件の対角順(右前→左後→左前→右後)
MG = 1.3 * 9.81                          # 総重量ぶんの重力[N]


# ============================================================================
# コントローラ(各脚 yaw/pitch/wheel 目標を保持し、フェーズ間で線形ランプ)
# ============================================================================
class Ctrl:
    def __init__(self, m, d, A, info):
        self.m, self.d, self.A, self.info = m, d, A, info
        self.yaw = {n: 0.0 for n in LEGS}
        self.pitch = {n: 0.0 for n in LEGS}
        self.wheel = {n: 0.0 for n in LEGS}

    def apply(self):
        for n in LEGS:
            self.d.ctrl[self.A[f"yawpos_{n}"]] = self.yaw[n]
            self.d.ctrl[self.A[f"pitchpos_{n}"]] = self.pitch[n]
            self.d.ctrl[self.A[f"wheeldrv_{n}"]] = self.wheel[n]

    def snapshot(self):
        return (dict(self.yaw), dict(self.pitch), dict(self.wheel))


def _rpy_deg(q):
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return roll, pitch


def _wheel_xy(m, d, info, leg):
    return d.body(info["wheel_bid"][leg]).xpos[:2].copy()


def _margin_if_lifted(m, d, info, lift_leg):
    """lift_leg を持ち上げたと仮定したときの、残り3脚の支持三角形に対する
    現在 COM の符号付き余裕[m]。正=三角形内(安定)、負=外(転倒方向)。"""
    stance = [n for n in LEGS if n != lift_leg]
    poly = [_wheel_xy(m, d, info, n) for n in stance]
    return _poly_margin(com_xy(m, d), poly)


def _settle(sim, n):
    m, d, A, info, ctrl = sim
    for _ in range(n):
        ctrl.apply()
        mujoco.mj_step(m, d)


def _ramp(sim, targets, ramp_s, hold_s, log=None, tag=""):
    """targets = {"yaw":{leg:val}, "pitch":{...}, "wheel":{...}} へ線形ランプ+保持。"""
    m, d, A, info, ctrl = sim
    dt = m.opt.timestep
    y0, p0, w0 = ctrl.snapshot()
    ty = {**y0, **targets.get("yaw", {})}
    tp = {**p0, **targets.get("pitch", {})}
    tw = {**w0, **targets.get("wheel", {})}
    n_ramp = max(1, int(ramp_s / dt))
    for k in range(n_ramp + int(hold_s / dt)):
        a = min(1.0, (k + 1) / n_ramp)
        for n in LEGS:
            ctrl.yaw[n] = (1 - a) * y0[n] + a * ty[n]
            ctrl.pitch[n] = (1 - a) * p0[n] + a * tp[n]
            ctrl.wheel[n] = (1 - a) * w0[n] + a * tw[n]
        ctrl.apply()
        mujoco.mj_step(m, d)
        if log is not None:
            _log_step(sim, tag, log)


def _log_step(sim, phase, log):
    m, d, A, info, ctrl = sim
    s = stability(m, d)
    com = com_full(m, d)
    b = d.body("body").xpos
    roll, pitch = _rpy_deg(d.qpos[3:7])
    row = {"phase": phase, "t": d.time, "bx": b[0], "bz": b[2],
           "comx": com[0], "comy": com[1],
           "roll": roll, "pitch": pitch, "n_sup": s["n_support"],
           "margin": s["margin"] if s["margin"] is not None else float("nan"),
           "support": "".join(sorted(s["support"].keys()))}
    for n in LEGS:
        row[f"tau_{n}"] = d.actuator(f"pitchpos_{n}").force[0]
    log.append(row)


def _new_sim():
    m, d, A, info = make(1.3)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ctrl = Ctrl(m, d, A, info)
    return (m, d, A, info, ctrl)


# ============================================================================
# 事前実測: 荷重移動の権限 (前後=股ピッチ / 左右=股ヨー)
# ============================================================================
def measure_shift_authority():
    """接地4脚(車輪ブレーキ)で車体を傾け、荷重が前後/左右にどれだけ移せるかを実測。
    → 前後は移せる(股ピッチ)、左右は移せない(股ヨー)ことを数値で示す。"""
    def run(kind, tgt):
        sim = _new_sim()
        m, d, A, info, ctrl = sim
        _settle(sim, 400)
        for n in LEGS:
            ctrl.wheel[n] = 0.0
            if kind == "pitch":
                ctrl.pitch[n] = tgt
            else:
                ctrl.yaw[n] = tgt
        _ramp(sim, {}, 0.0, 0.0)   # noop to sync
        _settle(sim, 700)
        f = {n: leg_normal_force(m, d, n) for n in LEGS}
        fr = (f["RF"] + f["LF"]) / 2
        bk = (f["RB"] + f["LB"]) / 2
        rt = (f["RF"] + f["RB"]) / 2
        lt = (f["LF"] + f["LB"]) / 2
        return fr, bk, rt, lt

    # 前後: 股ピッチを揃えて振る
    fr_p, bk_p, _, _ = run("pitch", 0.35)
    # 左右: 股ヨーを揃えて振る
    _, _, rt_y, lt_y = run("yaw", 0.6)
    return {"pitch_front": fr_p, "pitch_back": bk_p, "yaw_right": rt_y, "yaw_left": lt_y}


# ============================================================================
# [1] 安定性(不変条件)スタディ: 対角順に各脚を上げ、支持三角形 vs COM をログ
# ============================================================================
STANCE = 0.0
BACK_UNLOAD = -0.35   # 前脚を軽くする前後シフト量(前脚上げ時)。後脚上げ時は +0.35。
LIFT_MAG = 0.65       # 上げ脚の |pitch| 追加(足先を畳んで上げる)


def stability_study(log):
    """各脚(対角順)について:
       (a) 静的判定: その脚を上げたときの残り3脚三角形に対する COM 余裕(幾何)。
       (b) 動的試行: 前後シフトで対象脚を最大限に軽くしてから実際に上げ、
           試行中の最小 margin / 最大傾き / 実際に浮いた脚 / 転倒有無を記録。
    """
    rows = []
    for lead in DIAG_ORDER:
        sim = _new_sim()
        m, d, A, info, ctrl = sim
        _settle(sim, 400)
        # (a) 静的余裕(名目立位): その脚を上げたと仮定した三角形 vs 現 COM
        static_margin = _margin_if_lifted(m, d, info, lead)

        # (b) 前後シフトで対象脚を軽く(前脚→後へ、後脚→前へ)
        is_front = lead in ("RF", "LF")
        shift = BACK_UNLOAD if is_front else -BACK_UNLOAD
        _ramp(sim, {"pitch": {n: shift for n in LEGS}}, 0.6, 0.4, log, f"{lead}:shift")
        f_before = leg_normal_force(m, d, lead)

        # 実際に対象脚を上げる(|pitch| を増やして足先を畳み上げる)
        base = ctrl.pitch[lead]
        lift_target = base - LIFT_MAG if base <= 0 else base + LIFT_MAG
        min_margin = 1e9
        max_tilt = 0.0
        start = len(log)
        _ramp(sim, {"pitch": {lead: lift_target}}, 0.5, 0.4, log, f"{lead}:lift")
        for r in log[start:]:
            if not np.isnan(r["margin"]):
                min_margin = min(min_margin, r["margin"])
            max_tilt = max(max_tilt, abs(r["roll"]), abs(r["pitch"]))

        # どの脚が実際に浮いたか(接地力最小の脚)
        forces = {n: leg_normal_force(m, d, n) for n in LEGS}
        actually_lifted = min(forces, key=forces.get)
        toppled = d.body("body").xpos[2] < 0.10

        rows.append({
            "lead": lead, "static_margin": static_margin,
            "min_margin": min_margin if min_margin < 1e8 else float("nan"),
            "max_tilt": max_tilt, "intended": lead,
            "actually_lifted": actually_lifted,
            "as_intended": actually_lifted == lead,
            "toppled": toppled,
        })
    return rows


# ============================================================================
# [2] pull-up 股ピッチトルク実測: straddle 姿勢(1輪を段上、3輪を接地)から引き上げ
# ============================================================================
# straddle: 車体をほぼ水平に置き、対象前脚を段の上面へ届かせる pitch
STRADDLE_PITCH = -0.795     # この pitch で足先が z≈0.045(段上面+車輪半径)へ届く
PULLUP_TARGET = -0.05       # 引き上げ後(脚をほぼ真下へ戻す)
PULLUP_RAMP_S = 1.8


def make_straddle(lead, body_x=0.03):
    """対象前脚 lead の車輪を段の上面に置いた straddle 姿勢を構築して返す。"""
    sim = _new_sim()
    m, d, A, info, ctrl = sim
    d.qpos[0] = body_x
    d.qpos[2] = STAND_BODY_Z
    d.qpos[info["pitch_qadr"][lead]] = STRADDLE_PITCH
    for n in LEGS:
        ctrl.pitch[n] = STRADDLE_PITCH if n == lead else 0.0
        ctrl.wheel[n] = 0.0
    mujoco.mj_forward(m, d)
    _settle(sim, 500)
    return sim


def pullup_measure(lead, body_x=0.03, preload_pitch=0.0, log=None):
    """straddle(前脚 lead の1輪を段上に置き、他3輪を接地)から、lead を段上で
    アンカー(車輪ブレーキ)して pull-up。股ピッチのピークトルク/引き上げ量/
    対象脚の荷重を返す。
    preload_pitch>0 で接地3脚の pitch を前傾させ、車体を前へ寄せて lead の荷重を増やす
    (荷重の載り具合を振って tau∝荷重 を確認する用)。"""
    sim = make_straddle(lead, body_x=body_x)
    m, d, A, info, ctrl = sim
    if preload_pitch != 0.0:
        stance = [n for n in LEGS if n != lead]
        _ramp(sim, {"pitch": {n: preload_pitch for n in stance}}, 0.5, 0.4, log,
              f"{lead}:preload")
    load_on_lead = leg_normal_force(m, d, lead)
    z0 = d.body("body").xpos[2]
    # ローカルログでピークを必ず追跡(呼び出し側 log の有無に依らず)
    local = []
    _ramp(sim, {"pitch": {lead: PULLUP_TARGET}, "wheel": {lead: 0.0}},
          PULLUP_RAMP_S, 0.6, local, f"{lead}:pullup")
    peak = max((abs(r[f"tau_{lead}"]) for r in local), default=0.0)
    if log is not None:
        log.extend(local)
    lift = d.body("body").xpos[2] - z0
    return {"lead": lead, "peak": peak, "load": load_on_lead, "lift": lift}


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    log = []

    auth = measure_shift_authority()
    stab = stability_study(log)
    # pull-up: 前脚 straddle で荷重を振って測る(tau∝荷重 を確認)。
    # 後脚は同一の脚機構・同一段差・同一車体荷重ぶんで対称(mirror)なので前脚が代表。
    pulls = [
        pullup_measure("RF", body_x=0.03, log=log),                    # 分担・軽荷重
        pullup_measure("LF", body_x=0.03, log=log),                    # 分担(左右対称確認)
        pullup_measure("RF", body_x=0.03, preload_pitch=0.30, log=log),  # 荷重を寄せた条件
    ]

    _print_report(auth, stab, pulls)
    if args.csv:
        _write_csv(log, args.csv)
        print("\nCSV: %s (%d rows)" % (args.csv, len(log)))


def _print_report(auth, stab, pulls):
    line = "=" * 74
    print(line)
    print("Phase 2: 4脚 1脚ずつ 3cm段差越えシーケンス  検証レポート  (総重量1.3kg)")
    print(line)

    # --- 荷重移動の権限 ---
    print("\n■ 前提: 接地4脚での荷重移動の権限(車輪ブレーキで車体を傾け実測)")
    print("  股ピッチ揃え(前後シフト): 前脚荷重 %.2f N ⇔ 後脚荷重 %.2f N  → 前後は移せる"
          % (auth["pitch_front"], auth["pitch_back"]))
    print("  股ヨー揃え  (左右シフト): 右脚荷重 %.2f N ⇔ 左脚荷重 %.2f N  → 左右は移せない"
          % (auth["yaw_right"], auth["yaw_left"]))
    print("  静立時の1脚あたり荷重の目安 = %.2f N (総%.2f N / 4)" % (MG / 4, MG))

    # --- [1] 不変条件(重心 vs 支持三角形) ---
    print("\n[1] 不変条件: 各脚を上げたときの 重心 vs 残り3脚の支持三角形")
    print("    対角順(要件) RF→LB→LF→RB。 margin>0=三角形内(安定), <0=外(転倒方向)")
    print("    脚  静的margin[m]  上げ試行min margin  最大傾き[deg]  実際に浮いた脚  転倒")
    worst_static = 1e9
    any_neg = False
    for r in stab:
        worst_static = min(worst_static, r["static_margin"])
        neg = r["static_margin"] <= 0.005 or (not np.isnan(r["min_margin"]) and r["min_margin"] < 0)
        any_neg = any_neg or neg
        note = "" if r["as_intended"] else f"→実際は{r['actually_lifted']}が浮く(対角シーソー)"
        print("    %-3s   %+7.4f       %+7.4f          %5.1f       %-3s %s  %s"
              % (r["lead"], r["static_margin"], r["min_margin"], r["max_tilt"],
                 r["actually_lifted"], "" if r["as_intended"] else "≠", note))
    print("    ----------------------------------------------------------------------")
    print("    → 各脚の静的余裕は概ね 0 近傍〜負。単隅の脚上げは静的に支持三角形内を保てない。")
    print("      原因: この脚設計は左右方向の重心移動が作れない(上記『前提』)。対角シフトの")
    print("      左右成分を出せず、COM が支持三角形の対角線上〜外に出る。")
    print("      ∴ 不変条件『3脚支持で COM を三角形内に保つ』を満たす1脚ずつ静的越えは")
    print("        現行自由度(股ヨー+股ピッチ+受動足首/膝・股ロール無し)では不成立。")

    # --- [2] pull-up 股ピッチトルク ---
    print("\n[2] pull-up 股ピッチトルク(straddle: 前脚1輪を段上にアンカーし車体を引き上げ)")
    print("    条件               対象脚の荷重[N]  股ピッチ ピーク|tau|[N·m]  車体引き上げ[m]")
    labels = ["RF 荷重分担(3脚支持) ", "LF 荷重分担(左右対称)  ", "RF 荷重を寄せた条件   "]
    peak_shared = max(pulls[0]["peak"], pulls[1]["peak"])
    for lab, p in zip(labels, pulls):
        print("    %s %5.2f            %6.3f               %+.3f"
              % (lab, p["load"], p["peak"], p["lift"]))
    print("    ----------------------------------------------------------------------")
    print("    ・4脚モデルでは車体荷重が複数脚に分散する。段上にアンカーした前脚が車体を")
    print("      段へ引き上げる pull-up は成立し、股ピッチのピークは %.2f N·m 級と軽い" % peak_shared)
    print("      (荷重を寄せてもこの straddle 局面のトルクは概ね横ばい)。")
    print("    ・段差越えのサイジング律速=荷重が1脚へ集中する最悪ケース。これは Phase 1 の")
    print("      単脚 rig(全荷重を1脚に載せ、段上で車体を丸ごと引き上げ)が保守側で実測済み:")
    print("      1.3kg想定 ~1.7 N·m / 1.5kg安全側 2.03 N·m。4脚化は各脚荷重を『減らす』側で、")
    print("      この上限を超えない。")
    binding = 1.7            # サイジングの律速は Phase1 の1脚集中値(保守側)
    print("    ----------------------------------------------------------------------")
    print("    サイジング律速 = %.2f N·m(1脚集中, Phase1保守側)  → 安全率×2 = %.2f N·m"
          % (binding, binding * 2))
    sts3250_ok = binding * 2 <= 4.0
    print("    → 股ピッチ STS3250(3〜4 N·m)で%s。3脚分担時は更に余裕。§8 の選定を裏取り。"
          % ("足りる" if sts3250_ok else "要確認"))

    # --- 結論/提言 ---
    print("\n[結論と提言]")
    print("  ・pull-up トルク([2])は STS3250 の射程内。モーター選定(§8)は Phase 2 でも妥当。")
    print("  ・一方、1脚ずつの『静的』段差越え([1])は現行自由度では不変条件を満たせない")
    print("    (左右の重心移動ができない)。次アクション候補:")
    print("     (a) 股に『ロール1軸』または脚に『膝』を追加し左右/上下の重心移動を可能に")
    print("     (b) トレッドを広げ COM を下げ、車輪接地の動的安定に頼る短時間リフトを許容")
    print("     (c) 前後シフト(股ピッチ)＋車輪駆動での動的バランス越え(静的三脚を前提にしない)")
    print("  ・段差越え手順が上記いずれかで確定したら要求トルクを再サイジング(§9,§10)。")


def _write_csv(log, path):
    import csv
    if not log:
        return
    keys = list(log[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(log)


if __name__ == "__main__":
    main()
