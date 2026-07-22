"""
tachikoma / Phase 2  ―  転倒判定インスツルメント (重心 vs 支持多角形)

不変条件の裏取り用: 越えている1脚以外の3脚が接地を保ち、重心(COM)がその接地脚が
作る支持多角形(3脚なら三角形)の内側に、余裕をもって留まっているかを毎ステップ測る。

  - COM   : ロボット全体の重心。 d.subtree_com[body] (ルート body のサブツリー重心)。
  - 接地脚: ホイールが floor / step と接触している脚。 接触点(x,y) を集める。
  - margin: COM(x,y) から支持多角形の各辺までの符号付き距離の最小値。
            正 = 内側(余裕[m])、負 = 外側(転倒方向)。
"""
import numpy as np
import mujoco


def grounded_wheels(m, d):
    """接触している脚名 -> 接触点(x,y) list。 floor(id) と step(id) を対地とみなす。"""
    floor_id = m.geom("floor").id
    step_id = m.geom("step").id
    ground = {floor_id, step_id}
    # wheel geom id -> leg name
    wheelg = {}
    for i in range(m.ngeom):
        nm = m.geom(i).name
        if nm.startswith("wheelg_"):
            wheelg[i] = nm.split("_", 1)[1]
    contacts = {}
    for c in range(d.ncon):
        con = d.contact[c]
        g1, g2 = con.geom1, con.geom2
        leg = None
        if g1 in wheelg and g2 in ground:
            leg = wheelg[g1]
        elif g2 in wheelg and g1 in ground:
            leg = wheelg[g2]
        if leg is not None:
            contacts.setdefault(leg, []).append(con.pos[:2].copy())
    return contacts


def leg_normal_force(m, d, leg):
    """脚 leg の車輪が対地から受ける法線接触力の合計[N]。荷重の載り具合の指標。"""
    gid = m.geom(f"wheelg_{leg}").id
    f6 = np.zeros(6)
    tot = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        if con.geom1 == gid or con.geom2 == gid:
            mujoco.mj_contactForce(m, d, c, f6)
            tot += f6[0]          # 接触フレームの法線成分
    return tot


def com_xy(m, d):
    bid = m.body("body").id
    return d.subtree_com[bid][:2].copy()


def com_full(m, d):
    bid = m.body("body").id
    return d.subtree_com[bid].copy()


def _poly_margin(pt, poly):
    """凸多角形 poly(list of (x,y)) に対する点 pt の符号付き余裕[m]。
    正=内側。 多角形は反時計/時計どちらでも可(符号の一貫性で内外判定)。"""
    n = len(poly)
    if n == 0:
        return None
    if n == 1:
        return -np.linalg.norm(pt - poly[0])           # 1点支持=常に不安定
    if n == 2:
        # 線分支持: 線からの距離を負(不安定)として返す
        a, b = poly[0], poly[1]
        ab = b - a
        L = np.linalg.norm(ab) + 1e-12
        d_perp = abs(np.cross(ab, pt - a)) / L
        return -d_perp
    # 3点以上: 凸包を作り、各辺の符号付き距離の最小
    hull = _convex_hull(poly)
    # 多角形の向き(符号)を面積で決める
    signs = []
    dists = []
    k = len(hull)
    for i in range(k):
        a = hull[i]
        b = hull[(i + 1) % k]
        e = b - a
        L = np.linalg.norm(e) + 1e-12
        # 左手側を正とする符号付き距離
        s = np.cross(e, pt - a) / L
        signs.append(s)
        dists.append(s)
    # 全辺同符号なら内側。 margin = その符号方向の最小絶対距離。
    if all(x >= 0 for x in signs):
        return min(dists)
    if all(x <= 0 for x in signs):
        return min(-np.array(dists))
    # 符号が混在 = 外側。 最も外側(最大の負のはみ出し)を負で返す。
    return min(dists) if np.mean(signs) < 0 else -max(dists)


def _convex_hull(points):
    """2D 凸包 (Andrew's monotone chain)。 points: list of np.array([x,y])。"""
    pts = sorted([tuple(p) for p in points])
    if len(pts) <= 2:
        return [np.array(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [np.array(p) for p in hull]


def stability(m, d):
    """現在状態の安定性サマリを返す dict。
      com     : (x,y)
      support : {leg: (x,y)}  接地脚とその接触点(脚あたり平均)
      n_support : 接地脚数
      margin  : 支持多角形に対する COM の符号付き余裕[m] (正=安定)
      stable  : margin > 0 かつ接地脚>=3
    """
    contacts = grounded_wheels(m, d)
    support = {leg: np.mean(np.array(pts), axis=0) for leg, pts in contacts.items()}
    com = com_xy(m, d)
    poly = list(support.values())
    margin = _poly_margin(com, poly) if poly else None
    return {
        "com": com,
        "support": support,
        "n_support": len(support),
        "margin": margin,
        "stable": (margin is not None and margin > 0 and len(support) >= 3),
    }
