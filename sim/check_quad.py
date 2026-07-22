"""4脚モデル(Phase 2)の回帰チェック (uv run python sim/check_quad.py)。

Phase 2 の主要な挙動・知見が成立しているか無音で検証する:
  (1) 構成: 自由ベース6 + 脚16 自由度 / 能動12アクチュエータ / 総重量 ~1.3kg。
  (2) 立位: 自由ベースが4脚で静定し、全輪接地・COM が支持多角形の内側(margin>0)。
  (3) 前進: 全輪駆動で直進でき、転ばない。
  (4) 荷重移動の非対称: 股ピッチで前後荷重は移せる/股ヨーで左右荷重は移せない(知見の核)。
  (5) 単隅の脚上げは静的に支持三角形内を保てない(margin≈0以下 → 不変条件を満たせない)。
  (6) straddle pull-up: 前輪を段上にアンカーして車体を引き上げられ、股ピッチトルクは
      forcerange を飽和させない範囲。
"""
import numpy as np
import mujoco

from quad_model import make
from quad_stability import stability, leg_normal_force
from quad_sequence import (measure_shift_authority, stability_study,
                           pullup_measure, _margin_if_lifted, _new_sim,
                           _settle, LEGS, DIAG_ORDER)

# (1) 構成 --------------------------------------------------------------
m, d, A, info = make(1.3)
assert (m.nq, m.nv, m.nu) == (23, 22, 12), (m.nq, m.nv, m.nu)
assert m.ntendon == 0, "テンドン不採用のはず"
assert abs(m.geom("wheelg_RF").size[0] - 0.015) < 1e-6, "ホイール半径は0.015固定"
total = m.body_subtreemass[info["body_bid"]]
assert abs(total - 1.3) < 1e-3, ("総重量は1.3kg狙い", total)

# (2) 立位: 4脚で静定し全輪接地・margin>0 --------------------------------
sim = _new_sim()
m, d, A, info, ctrl = sim
_settle(sim, 1500)
s = stability(m, d)
assert s["n_support"] == 4, ("4輪接地のはず", s["n_support"])
assert s["margin"] > 0.03, ("COM は支持多角形の内側", s["margin"])
assert d.body("body").xpos[2] > 0.14, ("立位車体高", d.body("body").xpos[2])

# (3) 前進: 直進して転ばない --------------------------------------------
x0 = d.body("body").xpos[0]
for n in LEGS:
    ctrl.wheel[n] = 6.0
for _ in range(1200):
    ctrl.apply(); mujoco.mj_step(m, d)
assert d.body("body").xpos[0] - x0 > 0.05, "前進できるはず"
assert abs(d.body("body").xpos[1]) < 0.02, "直進(横ずれ小)"
assert d.body("body").xpos[2] > 0.12, "前進中に転ばない"

# (4) 荷重移動の非対称(知見の核) ---------------------------------------
auth = measure_shift_authority()
assert auth["pitch_front"] - auth["pitch_back"] > 1.0, \
    ("股ピッチで前後荷重を移せるはず", auth)
assert abs(auth["yaw_right"] - auth["yaw_left"]) < 0.5, \
    ("股ヨーでは左右荷重を移せない(ほぼ均等のまま)", auth)

# (5) 単隅の脚上げは静的に支持三角形内を保てない -------------------------
stab = stability_study([])
for r in stab:
    assert r["static_margin"] < 0.02, \
        ("単隅上げの静的余裕は概ね0以下のはず", r["lead"], r["static_margin"])
assert any(r["min_margin"] < 0 for r in stab if not np.isnan(r["min_margin"])), \
    "少なくとも一部の脚上げ試行で margin が負(支持三角形外)になる"

# (6) straddle pull-up: 引き上げ成立・トルク非飽和 -----------------------
frng = float(m.actuator("pitchpos_RF").forcerange[1])
p = pullup_measure("RF")
assert p["lift"] > 0.005, ("pull-up で車体が上がる", p["lift"])
assert p["peak"] < frng - 1e-3, ("pull-up トルクは forcerange を飽和させない", p["peak"], frng)
assert p["peak"] < 2.03, ("4脚分担の pull-up は Phase1 単脚集中(2.03)を超えない", p["peak"])

print("check_quad: OK")
print("  総重量            = %.4f kg" % total)
print("  立位 margin       = %.4f m (n_support=%d)" % (s["margin"], s["n_support"]))
print("  荷重移動 前後      = 前%.2f/後%.2f N (移せる)" % (auth["pitch_front"], auth["pitch_back"]))
print("  荷重移動 左右      = 右%.2f/左%.2f N (移せない)" % (auth["yaw_right"], auth["yaw_left"]))
print("  単隅上げ 静的margin = %s (≈0以下=不変条件を満たせない)"
      % ["%+.3f" % r["static_margin"] for r in stab])
print("  straddle pull-up   = 引上げ%.3fm, 股ピッチ|tau|=%.3f N·m (分担)"
      % (p["lift"], p["peak"]))
