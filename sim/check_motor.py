"""モーター選定(Phase 2b)の回帰チェック (uv run python sim/check_motor.py)。

全12軸の必要スペック実測が想定レンジに収まり、選定ロジックが崩れていないか無音検証:
  股ヨー : 据え切り/操舵スルーとも小(操舵は車体を持上げず、同軸操舵で慣性も極小)。
  ホイール: 連続/ピーク/牽引上限のトルクは極小、巡航速度に到達、必要回転数は妥当。
  股ピッチ: pull-up 律速 1.7 N·m を引継ぎ、×2 が STS3250 射程内。
"""
import numpy as np
import motor_sizing as MS

ys = MS.measure_yaw_scrub()
yl = MS.yaw_slew_torque(ys["scrub"])
wc = MS.measure_wheel_continuous()
tr = MS.measure_traction_ceiling()
cr = MS.confirm_cruise()

# 股ヨー: 据え切り・スルーとも軽負荷。慣性分は無視できる。
assert 0.005 < ys["scrub"] < 0.15, ("据え切りは小さいはず", ys["scrub"])
assert ys["load"] > 2.5, ("操舵前の接地荷重", ys["load"])
assert yl["T_inertia"] < 0.01, ("操舵スルーの慣性分は無視できる", yl["T_inertia"])
assert yl["T_slew"] * 2 < 3.0, ("股ヨー×2 は STS3215 射程に十分収まる", yl["T_slew"])

# ホイール: トルク極小 / 巡航到達 / 牽引上限は現実的
assert 0.002 < wc["cont"] < 0.02, ("登坂維持の連続トルクは極小", wc["cont"])
assert wc["nsup"] == 4, ("傾斜保持で4輪接地", wc["nsup"])
assert 0.01 < tr["ceiling"] < 0.06, ("牽引限界 µ·N·r は現実レンジ", tr["ceiling"])
assert cr["v"] > 0.4, ("巡航速度に到達できる", cr["v"])

# 必要回転数(運動学): 巡航 / 円周
rpm_c = MS.rpm(MS.OMEGA_CRUISE)
assert 300 < rpm_c < 340, ("巡航 0.5m/s で ~318 rpm", rpm_c)

# 股ピッチ: 引継ぎ律速の ×2 が STS3250(3〜4)射程内
assert MS.PITCH_PEAK * 2 <= 4.0, ("股ピッチ×2 は STS3250 射程内", MS.PITCH_PEAK)

print("check_motor: OK")
print("  股ヨー   据え切り=%.4f / 操舵スルー=%.4f N·m  (×2=%.3f → STS3215)"
      % (ys["scrub"], yl["T_slew"], max(ys["scrub"], yl["T_slew"]) * 2))
print("  ホイール 連続=%.4f / 牽引上限=%.4f N·m, 巡航到達=%.2f m/s, 必要=%.0f rpm"
      % (wc["cont"], tr["ceiling"], cr["v"], rpm_c))
print("  股ピッチ 律速=%.2f N·m (×2=%.2f → STS3250)" % (MS.PITCH_PEAK, MS.PITCH_PEAK * 2))
