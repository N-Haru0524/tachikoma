"""脚1本モデル(v2)の回帰チェック (uv run python sim/check_leg.py)。

要件の主要挙動が成立しているか無音で検証する:
  (1) 構成: 6自由度・3能動アクチュエータ (股ヨー/股ピッチ/ホイール)。膝なし・テンドン無し。
  (2) 足首サス: 荷重で受動的に沈む (コンプライアンス担保)。
  (3) 姿勢保持: 股ピッチを寝かせるほど保持トルクが単調に増える (最悪姿勢=ピーク)。
  (4) 段差 pull-up: 車輪を段上に置いた状態から股ピッチで車体を ~3cm 引き上げられる。
"""
import mujoco
from torque_probe import make, settle, scenario_hold, scenario_pullup

# (1) 構成 --------------------------------------------------------------
m, d, A = make(1.5)
assert (m.nq, m.nv, m.nu) == (6, 6, 3), (m.nq, m.nv, m.nu)
assert set(A) == {"hip_yaw_pos", "hip_pitch_pos", "wheel_drive"}, set(A)
assert m.ntendon == 0, "v2 はテンドン不採用のはず"
assert abs(m.geom("wheel_geom").size[0] - 0.015) < 1e-6, "ホイール半径は0.015固定"

# (2) 足首サスが荷重で沈む -----------------------------------------------
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[A["hip_yaw_pos"]] = 0.0
d.ctrl[A["hip_pitch_pos"]] = 0.30
settle(m, d, 3000)
sag = d.qpos[m.joint("ankle_sus").qposadr[0]]
assert sag > 0.001, ("ankle suspension should compress under load", sag)

# (3) 姿勢保持トルクが姿勢に対して単調増加 (1.5kg) -----------------------
rows, peak = scenario_hold(1.5)
taus = [abs(r[2]) for r in rows]
assert taus[-1] > taus[0], ("hold torque should grow with reach", taus[0], taus[-1])
assert all(b >= a - 5e-3 for a, b in zip(taus, taus[1:])), ("monotonic", taus)
assert abs(peak[2]) > 1.0, ("1.5kg hold peak should be ~1.5 N*m", peak)

# (4) 3cm段差 pull-up が成立 --------------------------------------------
traj, speak, speak_t, placed, climbed, lift = scenario_pullup(1.5)
assert placed, "車輪が段の上面に置けていない"
assert climbed, ("pull-up で車体が3cm上がるべき", lift)
assert abs(speak) < 20.0, ("pull-up torque must not saturate forcerange", speak)
assert abs(speak) > abs(peak[2]), ("段差pull-upが保持より重い律速のはず", speak, peak[2])

print("check_leg(v2): OK")
print("  ankle sag under load   = %.4f m" % sag)
print("  hold torque peak(1.5kg)= %.3f N*m at pitch %+.3f" % (abs(peak[2]), peak[1]))
print("  3cm step pull-up       = 成功, 引き上げ %.3f m, ピッチピーク %.3f N*m" % (lift, abs(speak)))
