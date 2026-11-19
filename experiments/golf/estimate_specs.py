"""제원 추정: 참조 차량(GTI, AV-21)으로 방법을 검증하고 Golf 에 적용 (docs/experiment_plan_golf.md 7절).

  python experiments/golf/estimate_specs.py   → experiments/golf/results/specs.json, specs.md

방법 (시험해 본 뒤 정한 것, 이유는 specs.md 에 기록)
  질량 m      : 직진 제동 구간 (구동력 거의 0). m*a = -F_brake - Cr0 - Cr2 v^2, 저항은 일반값으로 고정
  구동계 효율 : 직진 구동 구간 (제동 0). m 고정, m*a = eta*F_drive - Cr0 - Cr2 v^2
  4개 동시 회귀(m, eta, Cr0, Cr2)와 RLS 결과도 함께 기록 (Golf 에서는 값이 물리적으로 맞지 않음)
  lf          : 동시 비선형 회귀 결과를 기록하되 쓰지 않음 (GTI 에서 참값과 크게 달랐음) → 일반 Golf 무게 배분
  Iz          : lf 고정, |횡가속| < 5 m/s² 에서 회귀 → 초기값으로만 사용 (R1–R4 가 학습)
가속도는 상태를 2 Hz(종방향)·5 Hz(횡방향)로 거른 뒤 미분. 파일 앞뒤 2초는 버린다 (시작 순간의 속도 점프 때문).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_dynamics.golf import data as D  # noqa: E402
from deep_dynamics.golf import specs as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "experiments", "golf", "results")
REF = os.path.join(ROOT, "deep_dynamics", "reference_vehicles")
G = 9.81

KNOWN = dict(
    GTI=dict(m=1776.2, lf=1.194, lr=1.437, Iz=2763.49, Cr0=255.0, Cr2=0.3638, src="ladder configs.py GTI_THUNDERHILL (Stanford vehicles.py)"),
    AV21=dict(m=790.0, lf=1.248, lr=1.7328, src="deep_dynamics/cfgs/model/deep_dynamics_iac.yaml"),
)
# Golf 가정값 [추정]: 축거는 일반 Golf(Mk7) 제원 2.63 m, 앞바퀴굴림 해치백의 일반적인 앞 무게 몫 0.61.
# 저항은 일반 승용차 값: 굴림저항 계수 0.012, 공기저항 0.5*1.2*Cd 0.30*A 2.2 m² ≈ 0.40 N/(m/s)²
GOLF_L = 2.63
GOLF_FRONT_SHARE = 0.61
CR0_PER_KG = 0.012 * G
CR2_PRIOR = 0.40


def straight(vx, delta, r):
    return (vx > 5) & (np.abs(delta) < 0.02) & (np.abs(r) < 0.05)


def two_stage(v, a, F_brake, F_drive, cr2=CR2_PRIOR):
    """1단계: 제동 구간에서 m. 2단계: 구동 구간에서 eta. 저항은 일반값 (m 에 비례하는 굴림저항은 반복으로 맞춤).
    [변경 11/19] 제동 구간에 엔진 브레이크(음의 구동력)가 있는 표본도 넣고 그 힘을 eta·F_drive 로 식에 넣는다.
    m 과 eta 를 번갈아 맞춘다. 이전에는 |F_drive| < 300 N 인 제동 표본만 썼는데, 클러치 신호 수정(data.drive_force)으로
    엔진 브레이크가 드러나 그 조건을 만족하는 표본이 256 → 85 개로 줄었다."""
    br = (F_brake > 1000) & (F_drive < 300)
    dr = (F_brake < 1) & (F_drive > 500)
    m, eta = 1500.0, 1.0
    for _ in range(20):
        res = CR0_PER_KG * m + cr2 * v[br] ** 2
        # m*a = eta*F_drive - F_brake - res  →  m = Σ a*y / Σ a²
        y = -F_brake[br] + eta * F_drive[br] - res
        m = float((a[br] @ y) / (a[br] @ a[br]))
        y = m * a[dr] + CR0_PER_KG * m + cr2 * v[dr] ** 2
        eta = float((F_drive[dr] @ y) / (F_drive[dr] @ F_drive[dr]))
    resist = lambda k: CR0_PER_KG * m + cr2 * v[k] ** 2
    rms_b = float(np.sqrt(np.mean((-F_brake[br] + eta * F_drive[br] - resist(br) - m * a[br]) ** 2)))
    rms_d = float(np.sqrt(np.mean((m * a[dr] + resist(dr) - eta * F_drive[dr]) ** 2)))
    # 단순 비율의 분포 (진단용)
    big = br & (np.abs(a) > 0.5)
    ratio_b = np.median(-F_brake[big] / a[big]) if big.sum() else np.nan
    return dict(m=m, eta=eta, n_brake=int(br.sum()), n_drive=int(dr.sum()), rms_brake_N=rms_b, rms_drive_N=rms_d,
                Cr0_assumed=CR0_PER_KG * m, Cr2_assumed=cr2, median_Fbrake_over_a=float(ratio_b))


# ---------------------------------------------------------------- GTI
def gti():
    V = dict(ux=[], uy=[], r=[], delta=[], fxf=[], ax=[])
    L = dict(ux=[], uy=[], r=[], delta=[], vyd=[], rd=[])
    for f in sorted(glob.glob(os.path.join(REF, "gti", "exp_*.npz"))):
        d = np.load(f)
        fs = 100
        sm = {k: S.smooth(d[k].astype(float), 2.0, fs) for k in ["ux", "uy", "r", "delta", "fxf"]}
        sm["ax"] = S.deriv(sm["ux"], fs)
        for k in V:
            V[k].append(sm[k][2 * fs:-2 * fs:4])
        sl = {k: S.smooth(d[k].astype(float), 5.0, fs) for k in ["ux", "uy", "r", "delta"]}
        sl["vyd"], sl["rd"] = S.deriv(sl["uy"], fs), S.deriv(sl["r"], fs)
        for k in L:
            L[k].append(sl[k][2 * fs:-2 * fs:4])
    V = {k: np.concatenate(v) for k, v in V.items()}
    L = {k: np.concatenate(v) for k, v in L.items()}
    k = KNOWN["GTI"]
    fxr = np.where(V["fxf"] < 0, V["fxf"] * (1 - 0.66) / 0.66, 0.0)  # Stanford 데이터 생성 코드의 제동 배분
    st = straight(V["ux"], V["delta"], V["r"])
    a_eff = V["ax"] - V["uy"] * V["r"]
    out = dict(known=k)
    # (가) 전체 직진 구간, 바퀴 힘 합 = fxf + fxr (eta 없음)
    full = S.longitudinal_fit(V["ux"][st], V["uy"][st], V["r"][st], V["ax"][st], (V["fxf"] + fxr)[st])
    full.pop("rls_hist")
    out["lon_joint"] = full
    # (나) Golf 와 같은 2단계 (GTI 에는 구동·제동이 따로 없어 fxf 부호로 나눔)
    Fb = np.where(V["fxf"] < 0, -(V["fxf"] + fxr), 0.0)
    Fd = np.where(V["fxf"] > 0, V["fxf"], 0.0)
    out["lon_two_stage"] = two_stage(V["ux"][st], a_eff[st], Fb[st], Fd[st])
    ay = L["vyd"] + L["ux"] * L["r"]
    m5 = (L["ux"] > 8) & (np.abs(ay) < 5)
    args = (L["ux"][m5], L["uy"][m5], L["r"][m5], L["delta"][m5], L["vyd"][m5], L["rd"][m5])
    out["lat_free"] = S.lateral_fit(k["m"], k["lf"] + k["lr"], *args)
    out["lat_fixed_lf"] = S.lateral_fit(k["m"], k["lf"] + k["lr"], *args, lf0=k["lf"], Iz0=2000.0, fix_lf=True)
    out["lat_fixed_lf_ay3"] = S.lateral_fit(k["m"], k["lf"] + k["lr"], *[z[np.abs(ay[m5]) < 3] for z in args], lf0=k["lf"], Iz0=2000.0, fix_lf=True)
    return out


# ---------------------------------------------------------------- AV-21
def av21():
    L = dict(ux=[], uy=[], r=[], delta=[], vyd=[], rd=[])
    for f in sorted(glob.glob(os.path.join(REF, "av21", "*.csv"))):
        a = np.genfromtxt(f, delimiter=",", skip_header=1)
        fs = 25
        t = a[:, 0]
        ok = np.r_[True, np.abs(np.diff(t) - 0.04) < 1e-3]
        # 연속 구간마다 (필터는 끊긴 곳을 넘지 않게)
        bounds = np.r_[0, np.where(~ok)[0], len(t)]
        for i0, i1 in zip(bounds[:-1], bounds[1:]):
            if i1 - i0 < 200:
                continue
            seg = a[i0:i1]
            sl = {k: S.smooth(seg[:, c], 5.0, fs) for k, c in [("ux", 3), ("uy", 4), ("delta", 6), ("r", 7)]}
            sl["vyd"], sl["rd"] = S.deriv(sl["uy"], fs), S.deriv(sl["r"], fs)
            for k in L:
                L[k].append(sl[k][50:-50])
    L = {k: np.concatenate(v) for k, v in L.items()}
    k = KNOWN["AV21"]
    ay = L["vyd"] + L["ux"] * L["r"]
    m5 = (L["ux"] > 8) & (np.abs(ay) < 5)
    args = (L["ux"][m5], L["uy"][m5], L["r"][m5], L["delta"][m5], L["vyd"][m5], L["rd"][m5])
    return dict(known=k, lat_free=S.lateral_fit(k["m"], k["lf"] + k["lr"], *args),
                note="스로틀 %와 브레이크 압력만 있어 질량 검증 불가. 횡·요 식의 lf 만 검증")


# ---------------------------------------------------------------- Golf
def golf():
    cols = ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "F_drive", "F_brake_f", "F_brake_r", "grade_rad"]
    A = {c: [] for c in cols + ["ax"]}
    Lc = {c: [] for c in ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "vyd", "rd"]}
    lag = 20  # 힘이 가속도보다 100 ms 앞섬 [확인: 상관이 가장 큰 시간 차]
    sp = D.get_split()
    for n in sp["train"]:  # 학습 파일만 사용 (시험 파일 정보가 제원에 섞이지 않게)
        df = D.read_raw(n)
        f = {c: S.smooth(df[c].to_numpy(float), 2.0, 200) for c in cols}
        f["ax"] = S.deriv(f["Vx_mps"], 200)
        for c in ["F_drive", "F_brake_f", "F_brake_r"]:
            f[c] = np.r_[np.full(lag, f[c][0]), f[c][:-lag]]
        for c in A:
            A[c].append(f[c][400:-400:8])
        g = {c: S.smooth(df[c].to_numpy(float), 5.0, 200) for c in ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad"]}
        g["vyd"], g["rd"] = S.deriv(g["Vy_mps"], 200), S.deriv(g["yawRate_radps"], 200)
        for c in Lc:
            Lc[c].append(g[c][400:-400:8])
    A = {c: np.concatenate(v) for c, v in A.items()}
    Lc = {c: np.concatenate(v) for c, v in Lc.items()}
    st = straight(A["Vx_mps"], A["roadWheelAngle_rad"], A["yawRate_radps"])
    Fb = A["F_brake_f"] + A["F_brake_r"]
    out = dict(lag_ms=lag * 5, files=sp["train"])
    for gs, name in [(0, "no_grade"), (1, "grade_plus"), (-1, "grade_minus")]:
        a_eff = A["ax"] - A["Vy_mps"] * A["yawRate_radps"] + gs * G * np.sin(A["grade_rad"])
        out[f"two_stage_{name}"] = two_stage(A["Vx_mps"][st], a_eff[st], Fb[st], A["F_drive"][st])
        j = S.longitudinal_fit(A["Vx_mps"][st], A["Vy_mps"][st], A["yawRate_radps"][st], A["ax"][st], -Fb[st],
                               F_drive=A["F_drive"][st], grade=gs * A["grade_rad"][st] if gs else None)
        j.pop("rls_hist")
        out[f"joint_{name}"] = j
    # 2단계 결과 중 잔차가 작은 경사 부호를 채택
    best = min(["no_grade", "grade_plus", "grade_minus"], key=lambda k: out[f"two_stage_{k}"]["rms_brake_N"] + out[f"two_stage_{k}"]["rms_drive_N"])
    ts = out[f"two_stage_{best}"]
    out["chosen_grade"] = best
    m = ts["m"]
    lf = (1 - GOLF_FRONT_SHARE) * GOLF_L
    ay = Lc["vyd"] + Lc["Vx_mps"] * Lc["yawRate_radps"]
    m5 = (Lc["Vx_mps"] > 8) & (np.abs(ay) < 5)
    args = tuple(Lc[c][m5] for c in ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "vyd", "rd"])
    out["lat_free"] = S.lateral_fit(m, GOLF_L, *args)
    out["lat_fixed_lf"] = S.lateral_fit(m, GOLF_L, *args, lf0=lf, Iz0=2000.0, fix_lf=True)
    out["final"] = dict(m=round(m, 1), eta=round(ts["eta"], 3), L=GOLF_L, lf=round(lf, 3), lr=round(GOLF_L - lf, 3),
                        Iz=round(out["lat_fixed_lf"]["Iz"], 0), Cr0=round(ts["Cr0_assumed"], 1), Cr2=CR2_PRIOR,
                        Cf_lin=round(out["lat_fixed_lf"]["Cf"], 0), Cr_lin=round(out["lat_fixed_lf"]["Cr"], 0),
                        note="m·eta: 데이터 추정. lf: 가정(앞 무게 몫 0.61). Iz: lf 고정 회귀(초기값). 저항: 일반값")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(GTI=gti(), AV21=av21(), GOLF=golf())
    json.dump(res, open(os.path.join(OUT, "specs.json"), "w"), indent=1, default=float)
    g, t, v = res["GOLF"], res["GTI"], res["AV21"]
    pct = lambda e, k: f"{(e - k) / k * 100:+.1f}%"
    L = ["# 제원 추정 결과", "",
         "## 1. 방법 검증: GTI (제원을 아는 차)", "",
         "| 항목 | 참값 | 추정 | 차이 | 방법 |", "|---|---|---|---|---|",
         f"| 질량 (kg) | {t['known']['m']} | {t['lon_joint']['batch']['m']:.1f} | {pct(t['lon_joint']['batch']['m'], t['known']['m'])} | 직진 전 구간 동시 회귀 |",
         f"| 질량 (kg) | {t['known']['m']} | {t['lon_two_stage']['m']:.1f} | {pct(t['lon_two_stage']['m'], t['known']['m'])} | 2단계(제동 구간) — Golf 에 쓰는 방법 |",
         f"| 굴림저항 Cr0 (N) | {t['known']['Cr0']} | {t['lon_joint']['batch']['Cr0']:.1f} | — | 동시 회귀 |",
         f"| 공기저항 Cr2 | {t['known']['Cr2']} | {t['lon_joint']['batch']['Cr2']:.3f} | — | 동시 회귀 |",
         f"| lf (m) | {t['known']['lf']} | {t['lat_free']['lf']:.3f} | {pct(t['lat_free']['lf'], t['known']['lf'])} | 횡·요 동시 회귀 (lf 자유) |",
         f"| Iz (kg m²) | {t['known']['Iz']} | {t['lat_fixed_lf']['Iz']:.0f} | {pct(t['lat_fixed_lf']['Iz'], t['known']['Iz'])} | lf 를 참값으로 고정, \\|ay\\|<5 |",
         f"| Iz (kg m²) | {t['known']['Iz']} | {t['lat_fixed_lf_ay3']['Iz']:.0f} | {pct(t['lat_fixed_lf_ay3']['Iz'], t['known']['Iz'])} | lf 고정, \\|ay\\|<3 |",
         "",
         "## 2. 방법 검증: AV-21 (부분)", "",
         "| 항목 | 참값 | 추정 | 차이 |", "|---|---|---|---|",
         f"| lf (m) | {v['known']['lf']} | {v['lat_free']['lf']:.3f} | {pct(v['lat_free']['lf'], v['known']['lf'])} |",
         "",
         "## 3. Golf", "",
         f"- 경사 부호 선택: {g['chosen_grade']} (2단계 잔차 합이 가장 작음)",
         f"- 힘과 가속도의 시간 차 보정: {g['lag_ms']} ms", "",
         "| 경사 처리 | 질량 m (kg) | 효율 eta | 제동 행 수 | 구동 행 수 | 제동 잔차 (N) | 구동 잔차 (N) |", "|---|---|---|---|---|---|---|"]
    for nm in ["no_grade", "grade_plus", "grade_minus"]:
        q = g[f"two_stage_{nm}"]
        L.append(f"| {nm} | {q['m']:.0f} | {q['eta']:.3f} | {q['n_brake']} | {q['n_drive']} | {q['rms_brake_N']:.0f} | {q['rms_drive_N']:.0f} |")
    L += ["", "동시 회귀 (4개 계수 한꺼번에, 참고용):", "", "| 경사 처리 | eta | Cr0 | Cr2 | m |", "|---|---|---|---|---|"]
    for nm in ["no_grade", "grade_plus", "grade_minus"]:
        q = g[f"joint_{nm}"]["batch"]
        L.append(f"| {nm} | {q['eta']:.3f} | {q['Cr0']:.0f} | {q['Cr2']:.3f} | {q['m']:.0f} |")
    L += ["", f"횡·요 동시 회귀(lf 자유): lf {g['lat_free']['lf']:.3f} m, Iz {g['lat_free']['Iz']:.0f} — 쓰지 않음", "",
          "**최종 (configs.GOLF_THUNDERHILL 에 넣는 값)**", "", "| 항목 | 값 |", "|---|---|"]
    for k2, v2 in g["final"].items():
        L.append(f"| {k2} | {v2} |")
    open(os.path.join(OUT, "specs.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
