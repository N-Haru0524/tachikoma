"""
tachikoma / 無回転状態の向き検証 (uv run python sim/check_orientation.py)

初期姿勢の起点は「ベース(freejoint)を無回転 = identity quat(1,0,0,0) で置く」こと。
本スクリプトは、その無回転状態で **静定させずに**(=純粋な無回転のまま)モデルが正しい
起立方向になっているかを確認する:

  (1) ベース freejoint の quat が identity(=無回転)であること
  (2) ルート body の world 回転が単位行列(=world 軸と一致)であること
  (3) 車体(body)のローカル +z が world 上(+z)を向く(天地逆・横倒しでない。傾き<しきい値)
  (4) 幾何が起立: 車体が全車輪より上にあり、4輪とも車体より下にある

URDF/メッシュに向きの焼き付き(天地逆など)が無いことの回帰ガードになる。
make_stand() の静定(転倒し得る)とは切り離し、純粋な向きだけを見る。

使い方:
    uv run python sim/check_orientation.py                    # 既定ロボット
    uv run python sim/check_orientation.py --robot <name|path>
    uv run python sim/check_orientation.py --render out.png   # 図も保存(PIL があれば)
"""
import argparse
import numpy as np
import mujoco

import urdf_model as U

UP_TOL_DEG = 5.0            # 車体 up 軸が world 上からずれてよい許容角


def _free(m):
    fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    return fj, m.jnt_qposadr[fj], m.jnt_bodyid[fj]


def evaluate(robot=U.DEFAULT_ROBOT, base_z=0.25):
    """無回転状態を構築して評価結果 dict を返す(表示・アサートの両方で使う)。"""
    U.set_robot(robot)
    m, d = U.build()
    fj, qadr, base_bid = _free(m)
    # ベースを無回転(identity quat)で base_z に配置。settle しない=純粋な無回転状態。
    d.qpos[qadr:qadr + 7] = [0, 0, base_z, 1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    quat = d.qpos[qadr + 3:qadr + 7].copy()
    R_root = d.body(base_bid).xmat.reshape(3, 3)          # ルート world 回転(無回転なら I)
    R_body = d.body("body").xmat.reshape(3, 3)            # 車体(chassis)world 回転
    body_up = R_body[:, 2]                                # 車体ローカル +z の world 表現
    up_deg = float(np.degrees(np.arccos(np.clip(body_up[2], -1.0, 1.0))))
    body_z = float(d.body("body").xpos[2])
    wheel_z = {f: float(d.body(U.WHEEL_BODIES[f]).xpos[2]) for f in U.FEET}
    min_w, max_w = min(wheel_z.values()), max(wheel_z.values())

    checks = [
        ("ベース quat = identity(無回転)",
         bool(np.allclose(quat, [1, 0, 0, 0], atol=1e-6)),
         "quat=[%.4f %.4f %.4f %.4f]" % tuple(quat)),
        ("ルート回転 = 単位行列(world軸と一致)",
         bool(np.allclose(R_root, np.eye(3), atol=1e-4)),
         "max|R-I|=%.2e" % float(np.max(np.abs(R_root - np.eye(3))))),
        ("車体 up が world 上 (傾き<%.0f°)" % UP_TOL_DEG,
         up_deg < UP_TOL_DEG,
         "body_up=[%.3f %.3f %.3f] 傾き=%.2f°" % (body_up[0], body_up[1], body_up[2], up_deg)),
        ("車体が全車輪より上(=天地正)",
         body_z > max_w,
         "body_z=%.3f  wheel_z=[%.3f..%.3f]" % (body_z, min_w, max_w)),
    ]
    ok = all(c[1] for c in checks)
    return dict(ok=ok, checks=checks, quat=quat, up_deg=up_deg,
                body_z=body_z, wheel_z=wheel_z, robot=U.ROBOT_NAME,
                m=m, d=d, base_bid=base_bid)


def _view(m, d, interactive=False):
    """無回転状態を MuJoCo ビューアで表示。
    interactive=False: 物理を進めない(mj_step しない)ので重力で崩れず、その姿勢のまま
      カメラを回して天地・軸を目視確認できる。
    interactive=True : 重力オフ＋台座(ベース)を無回転位置に固定し、関節サーボだけを
      効かせる。ビューア右パネルの Control スライダ(pitchpos_*/yawpos_*/wheeldrv_*)を
      ドラッグすると、崩れず・台座も動かさずに各関節を動かして可動域や向きを確認できる。"""
    import time
    import mujoco.viewer
    fj, qadr, _ = _free(m)
    dofadr = m.jnt_dofadr[fj]
    base7 = d.qpos[qadr:qadr + 7].copy()               # 無回転の台座姿勢を保存
    if interactive:
        m.opt.gravity[:] = 0.0                          # 重力オフ=関節を自由に動かせる
        print("  ビューア起動(interactive): Control スライダで関節を動かせます")
        print("    pitchpos_* = 股ピッチ / yawpos_* = 股ヨー / wheeldrv_* = 車輪駆動")
        print("    重力オフ・台座固定なので崩れません。ウィンドウを閉じると終了。")
    else:
        print("  ビューア起動: 無回転状態を静止表示中(ウィンドウを閉じると終了)")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            if interactive:
                mujoco.mj_step(m, d)                    # サーボがスライダ目標へ関節を動かす
                d.qpos[qadr:qadr + 7] = base7           # 台座を無回転位置に固定
                d.qvel[dofadr:dofadr + 6] = 0.0
            mujoco.mj_forward(m, d)                     # 描画用に運動学を更新
            v.sync()
            time.sleep(m.opt.timestep if interactive else 0.02)


def _render(m, d, path):
    try:
        from PIL import Image
    except ImportError:
        print("  (--render 省略: PIL が無い。`uv run --with pillow ...` で実行すると保存できます)")
        return
    r = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, d.body("body").xpos[2]]
    cam.distance, cam.azimuth, cam.elevation = 0.9, 135, -18
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(path)
    print("  図を保存:", path)


def main():
    ap = argparse.ArgumentParser(description="無回転状態の向き検証")
    U.add_robot_arg(ap)
    ap.add_argument("--render", default=None, metavar="PNG",
                    help="無回転状態を PNG に描画保存(PIL があれば)")
    ap.add_argument("--view", action="store_true",
                    help="無回転状態を MuJoCo ビューアで静止表示(重力で崩さず向きを目視)")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="ビューアで関節を対話操作(重力オフ＋台座固定, Controlスライダで動かす)")
    args = ap.parse_args()

    r = evaluate(args.robot)
    print("=" * 62)
    print("無回転状態の向き検証  robot=%s" % r["robot"])
    print("=" * 62)
    for name, passed, detail in r["checks"]:
        print("  [%s] %-30s %s" % ("OK" if passed else "NG", name, detail))
    print("  車輪高: " + "  ".join("%s=%.3f" % (f, z) for f, z in r["wheel_z"].items()))
    print("-" * 62)
    print("  → %s" % ("★無回転状態は正常(天地・軸ともに正しい)★" if r["ok"]
                       else "✗ 向きに問題あり(URDF/メッシュの焼き付きを疑う)"))
    if args.render:
        _render(r["m"], r["d"], args.render)
    if args.view or args.interactive:
        _view(r["m"], r["d"], interactive=args.interactive)
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
