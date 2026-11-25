"""코너별 R1 vs R6 비교 (한계 9.0 m/s², 시드 0 주행 기록). 새 주행 없음.
코너 = 기록선 곡률 반지름 < 150 m 가 이어진 구간 (10 m 안 틈은 합침, 20 m 미만 버림), 진행 거리 기준.
코너마다: 걸린 시간(코너 앞 40 m ~ 뒤 40 m), 진입 속도(코너 시작), 최저 속도, 탈출 속도(코너 끝), 최대 |ay|, 최대 |횡오차|.
결과: realsim_out/corners.json + 표 출력"""
import json, os, sys
import numpy as np
sys.argv.insert(2, "run")
import drive_compare as DC, speed_profile as SP
tr, grade, info = DC.V.recorded_track(DC.V.lap_info("2023-10_5"))
x, y = tr.x.cpu().numpy(), tr.y.cpu().numpy()
k, ds = SP.curvature(x, y)
S = np.cumsum(ds) - ds[0]; Ltot = S[-1] + ds[-1]; i0 = info["start_idx"]
def prog_of_idx(idx):
    d = S[np.asarray(idx, int) % len(S)] - S[i0]
    return np.where(d < -50, d + Ltot, d)
P = prog_of_idx(np.arange(len(S))); o = np.argsort(P); Ps, Ks = P[o], np.abs(k[o])
tight = Ks > 1 / 150
corners, a = [], None
for j in range(len(Ps)):
    if tight[j] and a is None: a = j
    if (not tight[j] or j == len(Ps) - 1) and a is not None:
        corners.append([Ps[a], Ps[j - 1]]); a = None
m = []
for c in corners:
    if m and c[0] - m[-1][1] < 10: m[-1][1] = c[1]
    else: m.append(c)
corners = [c for c in m if c[1] - c[0] >= 20]
def load(sim, R):
    d = np.load(f"realsim_out/log_profile_{sim}_{R}_ay9.0.npz")
    p = prog_of_idx(d["idx"]); p = np.where(np.arange(len(p)) < 50, np.minimum(p, 60), p); p = np.maximum.accumulate(p)
    v = np.hypot(d["vx"], d["vy"]); ay = d["vx"] * d["r"]
    return dict(p=p, v=v, ay=ay, ey=d["ey"], plan=d["vref_plan"])
def at(L, s):  # 진행 거리 s 에서의 시각 (선형 보간)
    return float(np.interp(s, L["p"], np.arange(len(L["p"])) * 0.04))
out = {}
for sim in ("commonroad", "realsim"):
    Ls = {R: load(sim, R) for R in ("R1", "R6")}
    rows = []
    for ci, (s0, s1) in enumerate(corners):
        rmin = 1 / Ks[(Ps >= s0) & (Ps <= s1)].max()
        row = dict(corner=ci + 1, s0=round(float(s0)), s1=round(float(s1)), R_min=round(float(rmin)))
        for R, L in Ls.items():
            win = (L["p"] >= s0) & (L["p"] <= s1)
            row[R] = dict(t=at(L, min(s1 + 40, Ltot - 2)) - at(L, max(s0 - 40, 1)),
                          v_in=float(np.interp(s0, L["p"], L["v"])), v_min=float(L["v"][win].min()), v_out=float(np.interp(s1, L["p"], L["v"])),
                          ay_max=float(np.abs(L["ay"][win]).max()), ey_max=float(np.abs(L["ey"][win]).max()))
        rows.append(row)
    out[sim] = rows
    print(f"\n== {sim} (한계 9.0, 시드 0)   시간 차 = R6 − R1 (+ 면 R1 빠름)")
    print(" #   구간(m)     R_min | 시간차  | 진입 R1/R6  | 최저 R1/R6  | 탈출 R1/R6  | 최대ay R1/R6 | 최대|ey| R1/R6")
    for r in rows:
        a_, b_ = r["R1"], r["R6"]
        print(f"{r['corner']:2d} {r['s0']:5d}-{r['s1']:5d} {r['R_min']:5d} | {b_['t']-a_['t']:+6.2f} | {a_['v_in']:5.1f}/{b_['v_in']:5.1f} | {a_['v_min']:5.1f}/{b_['v_min']:5.1f} | {a_['v_out']:5.1f}/{b_['v_out']:5.1f} | {a_['ay_max']:5.1f}/{b_['ay_max']:5.1f} | {a_['ey_max']:4.2f}/{b_['ey_max']:4.2f}")
    tot1 = at(Ls["R1"], Ltot - 2) - at(Ls["R1"], 1); tot6 = at(Ls["R6"], Ltot - 2) - at(Ls["R6"], 1)
    inc = sum(r["R6"]["t"] - r["R1"]["t"] for r in rows)
    print(f"   전체 랩 시간 차 {tot6 - tot1:+.2f}s, 그중 코너(앞뒤 40 m 포함) {inc:+.2f}s")
    for R, L in Ls.items():
        st = np.ones(len(L["p"]), bool)
        for s0, s1 in corners: st &= ~((L["p"] >= s0 - 40) & (L["p"] <= s1 + 40))
        vp = np.interp(L["p"] % Ltot, P[o], L["plan"][o])
        print(f"   {R}: 코너 밖 평균 (계획 − 실제) 속도 {np.mean(vp[st] - L['v'][st]):+.2f} m/s, 코너 안 {np.mean(vp[~st] - L['v'][~st]):+.2f} m/s")
json.dump(dict(corners=[[float(a), float(b)] for a, b in corners], **out), open("realsim_out/corners.json", "w"), indent=1)
