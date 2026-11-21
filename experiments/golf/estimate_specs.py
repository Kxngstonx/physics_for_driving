"""Golf 차량 계수 총괄 (11/21 사용자 결정) + 제원 추정 방법 검증 (docs/experiment_plan_golf.md 7절, [결정 11/21]).

  python experiments/golf/estimate_specs.py           → deep_dynamics/golf/specs.json (커밋하는 입력 설정), results/specs_build.md
                                                         + 방법 검증 results/specs_validation.json·md (참고)
  python experiments/golf/estimate_specs.py recheck   → results/specs_recheck.* (11/21 질량·효율 재확인, 참고)

[변경 11/21] 모델(configs)·데이터(data)·시뮬레이터(sim)가 쓰는 차량 계수는 모두 build_specs() 가 만든 specs.json 에서 온다.
  출처 규칙: A 데이터로 정해지나 → B 공개값이 일관되나 → C 시기·차마다 바뀌나. A 아니오 + B 예 → 공개값, 그 밖 → 데이터.
  데이터로 정하는 값은 20개 파일 전체. 항목마다 source·basis 를 함께 적는다.
  아래 "방법" 은 11/18~11/20 의 이전 방법 설명이다 (질량 2단계·직선 강성은 지금 쓰지 않음 → 방법 검증 보고에만 남김).

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
LON_COLS = ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "F_drive", "F_brake_f", "F_brake_r", "grade_rad"]
LAG = 20  # 힘이 가속도보다 100 ms 앞섬 [확인: 상관이 가장 큰 시간 차]


def golf_lon_samples(name, extra_grade=False, **raw_kw):
    """한 파일 → 종방향 회귀용 표본 (2 Hz 평활, 힘은 LAG 만큼 늦춤, 앞뒤 2초 버림, 8칸마다).
    extra_grade: [11/21 재확인] 경사 두 가지를 더한다 (오르막 +, rad)
      grade_vu  = 관성 센서 위아래 속도 / 수평 속력 (0.5 Hz 2차 저역 통과, 11/18~11/20 시뮬레이터가 쓰던 경사)
      grade_osm = 트랙 경사 지도 (data.grade_at, track_osm 출력)
    raw_kw: data.read_raw 에 넘김 (gears, wheel_r, drive_eff). build_specs 가 새로 추정한 값으로 부를 때"""
    df = D.read_raw(name, **raw_kw)
    f = {c: S.smooth(df[c].to_numpy(float), 2.0, 200) for c in LON_COLS}
    f["ax"] = S.deriv(f["Vx_mps"], 200)
    for c in ["F_drive", "F_brake_f", "F_brake_r"]:
        f[c] = np.r_[np.full(LAG, f[c][0]), f[c][:-LAG]]
    if extra_grade:
        vu = D.lowpass(df["Vu_mps"].to_numpy(), cutoff=0.5, order=2)
        sp = np.hypot(df["Ve_mps"].to_numpy(), df["Vn_mps"].to_numpy())
        f["grade_vu"] = np.arctan2(vu, np.maximum(sp, 1.0))
        f["posE_m"], f["posN_m"] = df["posE_m"].to_numpy(float), df["posN_m"].to_numpy(float)
    out = {c: v[400:-400:8] for c, v in f.items()}
    if extra_grade:
        out["grade_osm"] = D.grade_at(out.pop("posE_m"), out.pop("posN_m"))
    return out


def golf():
    A = {c: [] for c in LON_COLS + ["ax"]}
    Lc = {c: [] for c in ["Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "vyd", "rd"]}
    lag = LAG
    sp = D.get_split()
    for n in sp["train"]:  # 학습 파일만 사용 (시험 파일 정보가 제원에 섞이지 않게)
        f = golf_lon_samples(n)
        for c in A:
            A[c].append(f[c])
        df = D.read_raw(n)
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


def validation_report():
    """방법 검증 (참고용): GTI·AV-21 에 추정 방법을 돌려 아는 값과 비교 + 이전 Golf 2단계 결과. specs.json 에는 쓰지 않는다.
    [11/21] 주의: GTI 의 fxf 는 저항을 이미 뺀 순 힘이라 질량·효율 검증에는 맞지 않는다 (계획서 [결정 11/21] 4)."""
    os.makedirs(OUT, exist_ok=True)
    res = dict(GTI=gti(), AV21=av21(), GOLF=golf())
    json.dump(res, open(os.path.join(OUT, "specs_validation.json"), "w"), indent=1, default=float)
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
          "**참고: 이전(11/18~11/20) 방법의 결과 — [11/21] 쓰지 않음. 쓰는 값은 deep_dynamics/golf/specs.json**", "", "| 항목 | 값 |", "|---|---|"]
    for k2, v2 in g["final"].items():
        L.append(f"| {k2} | {v2} |")
    open(os.path.join(OUT, "specs_validation.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


GRADES = ("grade_rad", "grade_vu", "grade_osm")


def recheck():
    """[11/21 사용자 승인] 질량·효율 재확인. 방법(직진 조건, 2단계 회귀, LAG)은 그대로 두고 비교만 더한다.
    경사 3가지(데이터 grade_rad 열 = 참고, 관성 센서 Vu, OSM 지도) × 파일 집합 2가지(학습 14, 전체 20) + 파일별 값.
    앞 제동 몫 0.658 → 0.75 처럼 일부 표본·잘못된 경사에 끌린 값인지 확인하려는 것. specs.json 은 건드리지 않는다.
    → results/specs_recheck.json, specs_recheck.md"""
    sp = D.get_split()
    names = D.available()
    per = {n: golf_lon_samples(n, extra_grade=True) for n in names}

    def fit(ns, grade):
        A = {c: np.concatenate([per[n][c] for n in ns]) for c in per[ns[0]]}
        st = straight(A["Vx_mps"], A["roadWheelAngle_rad"], A["yawRate_radps"])
        a_eff = A["ax"] - A["Vy_mps"] * A["yawRate_radps"] + (G * np.sin(A[grade]) if grade else 0.0)  # 오르막이면 중력이 감속
        Fb = A["F_brake_f"] + A["F_brake_r"]
        return two_stage(A["Vx_mps"][st], a_eff[st], Fb[st], A["F_drive"][st])

    res = dict(train=sp["train"], all=names, pooled={}, per_file={})
    for gr in (None,) + GRADES:
        key = gr or "no_grade"
        res["pooled"][key] = dict(train=fit(sp["train"], gr), all=fit(names, gr))
        res["per_file"][key] = {n: fit([n], gr) for n in names}
    # 경사 원천끼리 일치 정도 (직진 표본, 전체 파일)
    A = {c: np.concatenate([per[n][c] for n in names]) for c in ["Vx_mps", "roadWheelAngle_rad", "yawRate_radps"] + list(GRADES)}
    st = straight(A["Vx_mps"], A["roadWheelAngle_rad"], A["yawRate_radps"])
    res["grade_agreement"] = {f"{a}~{b}": dict(corr=float(np.corrcoef(A[a][st], A[b][st])[0, 1]),
                                               rms_deg=float(np.degrees(np.sqrt(np.mean((A[a][st] - A[b][st]) ** 2)))))
                              for a, b in (("grade_vu", "grade_osm"), ("grade_rad", "grade_vu"), ("grade_rad", "grade_osm"))}
    json.dump(res, open(os.path.join(OUT, "specs_recheck.json"), "w"), indent=1, default=float)

    q = lambda v: np.nanpercentile(v, [0, 25, 50, 75, 100])  # 제동 표본 0 인 파일(m 계산 안 됨)은 뺀다
    L = ["# 질량·구동계 효율 재확인 (11/21)", "", "방법은 specs.json 과 같다 (직진, 2단계 회귀). 바꾼 것: 경사 원천, 파일 집합. 기존 값: m 1496.9 kg, η 0.699 (grade_rad 비교 후 '경사 없음' 채택, 학습 파일).", "",
         "## 1. 모은 표본 (pooled)", "", "| 경사 | 파일 | m (kg) | η | 제동 표본 | 구동 표본 | 제동 잔차 (N) | 구동 잔차 (N) |", "|---|---|---|---|---|---|---|---|"]
    for key, d in res["pooled"].items():
        for fs, v in d.items():
            L.append(f"| {key} | {fs} | {v['m']:.0f} | {v['eta']:.3f} | {v['n_brake']} | {v['n_drive']} | {v['rms_brake_N']:.0f} | {v['rms_drive_N']:.0f} |")
    L += ["", "## 2. 파일별 (최소 / 25% / 중앙 / 75% / 최대)", "", "| 경사 | 대상 파일 | m (kg) | η |", "|---|---|---|---|"]
    for key, d in res["per_file"].items():
        for lab, keep in (("제동 표본 1개 이상", 1), ("제동 표본 20개 이상", 20)):
            vs = [v for v in d.values() if v["n_brake"] >= keep]
            L.append(f"| {key} | {lab} ({len(vs)}개) | {' / '.join(f'{x:.0f}' for x in q([v['m'] for v in vs]))} | "
                     f"{' / '.join(f'{x:.3f}' for x in q([v['eta'] for v in vs]))} |")
    L += ["", "파일별 값 (경사 = grade_vu):", "", "| 파일 | 분할 | m | η | 제동 표본 |", "|---|---|---|---|---|"]
    split_of = {n: k for k in ("train", "val", "test") for n in sp[k]}
    for n, v in res["per_file"]["grade_vu"].items():
        L.append(f"| {n} | {split_of.get(n, '-')} | {v['m']:.0f} | {v['eta']:.3f} | {v['n_brake']} |")
    L += ["", "## 3. 경사 원천끼리 일치 (직진 표본)", "", "| 비교 | 상관 | RMS 차이 (도) |", "|---|---|---|"]
    L += [f"| {k} | {v['corr']:.3f} | {v['rms_deg']:.3f} |" for k, v in res["grade_agreement"].items()]
    open(os.path.join(OUT, "specs_recheck.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


# ================================================================ 총괄: specs.json (11/21)
V_MOVE = 5.0  # 달리는 중 판정 (m/s)
CR0_COEF = 0.012  # 굴림 저항 계수 [가정: 일반 승용차]
CR2_ASSUMED = CR2_PRIOR  # 0.5·ρ·Cd·A = 0.5·1.2·0.30·2.2 ≈ 0.40 [가정: 일반 승용차]


def _per_file(fn, names):
    return [fn(D.read_csv(n)) for n in names]


def est_wheel_radius(names):
    """구동되지 않는 뒷바퀴: 차속 ÷ 뒷바퀴 회전 속도 (rad/s). 직진·제동 없음 표본의 중앙값."""
    def one(df):
        v, r = df["Vx_mps"].to_numpy(), df["yawRate_radps"].to_numpy()
        w = 0.5 * (df["wheelspeed_rl"].to_numpy() + df["wheelspeed_rr"].to_numpy())
        k = (v > V_MOVE) & (np.abs(r) < 0.05) & ((df["brake_rl_N"] + df["brake_rr_N"]).to_numpy() < 10.0) & (w > 1.0)
        return (v[k] / w[k])[::8]
    x = np.concatenate(_per_file(one, names))
    return float(np.median(x)), dict(n=int(len(x)), p25_p75=np.percentile(x, [25, 75]).tolist())


def est_gear_ratios(names, n_gears=4):
    """엔진 회전 ÷ 앞바퀴 평균 회전의 분포에서 가장 많이 몰린 n_gears 곳 (봉우리) → 봉우리 ±3% 안 표본의 중앙값."""
    from scipy.signal import find_peaks

    def one(df):
        wf = 0.5 * (df["wheelspeed_fl"].to_numpy() + df["wheelspeed_fr"].to_numpy())
        eng = df["engine_rpm"].to_numpy() * 2.0 * np.pi / 60.0
        k = (wf > 5.0) & (df["Vx_mps"].to_numpy() > V_MOVE)
        return (eng[k] / wf[k])[::4]
    x = np.concatenate(_per_file(one, names))
    x = x[(x > 1.5) & (x < 15.0)]
    h, e = np.histogram(x, bins=np.arange(1.5, 15.0, 0.02))
    pk, _ = find_peaks(h, distance=25)
    top = pk[np.argsort(h[pk])[::-1][:n_gears]]
    cen = 0.5 * (e[top] + e[top + 1])
    vals = sorted((float(np.median(x[np.abs(x / c - 1) < 0.03])) for c in cen), reverse=True)
    share = float(sum((np.abs(x / c - 1) < 0.03).mean() for c in cen))
    return vals, dict(n=int(len(x)), frac_within_3pct=share)


def est_brake(names):
    """앞 제동 몫: 속도 > 5 m/s · 네 바퀴 제동력 합 ≥ 2 kN 표본의 앞 몫 중앙값 (11/19 조건).
    제동 하한: 같은 속도 조건에서 앞 제동력 > 500 N 인 표본의 상위 1% (앞바퀴 힘으로는 음수)."""
    shares, front, per = [], [], []
    for n in names:
        df = D.read_csv(n)
        v = df["Vx_mps"].to_numpy()
        bf = (df["brake_fl_N"] + df["brake_fr_N"]).to_numpy()
        br = (df["brake_rl_N"] + df["brake_rr_N"]).to_numpy()
        k = (v > V_MOVE) & (bf + br >= 2000.0)
        sh = bf[k] / (bf[k] + br[k])
        shares.append(sh)
        per.append(float(np.median(sh)) if len(sh) else np.nan)
        front.append(bf[(v > V_MOVE) & (bf > 500.0)])
    sh, fr = np.concatenate(shares), np.concatenate(front)
    return float(np.median(sh)), -float(np.percentile(fr, 99)), dict(
        share_n=int(len(sh)), share_per_file_min_max=[float(np.nanmin(per)), float(np.nanmax(per))], front_n=int(len(fr)))


def est_steer_rate(names, q=99.9):
    """자동 조향 구간, 5 m/s 이상, 10 Hz 저역 통과 뒤 조향 속도 |dδ/dt| 의 99.9%."""
    def one(df):
        d = D.lowpass(df["roadWheelAngle_rad"].to_numpy(), cutoff=10.0)
        r = np.abs(np.gradient(d, 1.0 / D.FS))
        k = (df["Vx_mps"].to_numpy() > V_MOVE) & (df["auto_steer_bool"].to_numpy() > 0)
        return r[k]
    x = np.concatenate(_per_file(one, names))
    return float(np.percentile(x, q)), dict(n=int(len(x)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def est_ay_target(names, q=99.0):
    """실제 옆 가속도 dvy/dt + vx·r 의 상위 1% (calibrate_sim 의 마찰 보정 목표). 로더와 같은 처리:
    10 Hz 저역 통과 → 25 Hz, 정지(vx < V_STOP) 제외. 교차 확인: 차체 관성 센서 ayCG_mps2 (같은 필터)."""
    a, vr, cg = [], [], []
    for n in names:
        df = D.read_csv(n)
        f = D.lowpass(df[["Vx_mps", "Vy_mps", "yawRate_radps", "ayCG_mps2"]].to_numpy(float))[::D.EVERY]
        k = f[:, 0] >= D.V_STOP
        a.append((np.gradient(f[:, 1], D.DT) + f[:, 0] * f[:, 2])[k]); vr.append((f[:, 0] * f[:, 2])[k]); cg.append(f[k, 3])
    a, vr, cg = (np.abs(np.concatenate(x)) for x in (a, vr, cg))
    return float(np.percentile(a, q)), dict(vx_r_p99=float(np.percentile(vr, q)), ayCG_p99=float(np.percentile(cg, q)), n=int(len(a)))


def est_eta(names, m, cr0, cr2, gears, wheel_r):
    """효율: 질량을 공개값으로 고정하고 직진 구동 구간 회귀 (OSM 경사 포함). 역방향 회귀도 함께 (회귀 희석 확인)."""
    A = {c: [] for c in ("Vx_mps", "Vy_mps", "yawRate_radps", "roadWheelAngle_rad", "ax", "F_drive", "F_brake_f", "F_brake_r", "grade_osm")}
    for n in names:
        o = golf_lon_samples(n, extra_grade=True, gears=gears, wheel_r=wheel_r, drive_eff=1.0)
        for c in A:
            A[c].append(o[c])
    A = {c: np.concatenate(v) for c, v in A.items()}
    st = straight(A["Vx_mps"], A["roadWheelAngle_rad"], A["yawRate_radps"])
    k = st & (A["F_brake_f"] + A["F_brake_r"] < 1) & (A["F_drive"] > 500)
    a = A["ax"] - A["Vy_mps"] * A["yawRate_radps"] + G * np.sin(A["grade_osm"])
    y = m * a[k] + cr0 + cr2 * A["Vx_mps"][k] ** 2
    F = A["F_drive"][k]
    fwd, rev = float(F @ y / (F @ F)), float(y @ y / (F @ y))
    return fwd, dict(n=int(k.sum()), reverse=rev, rms_N=float(np.sqrt(np.mean((y - fwd * F) ** 2))))


def fiala(alpha, C, mu, Fz):
    """Fiala(브러시) 타이어 곡선 (연구실 논문과 같은 형식). alpha 미끄럼각, C 강성 (N/rad), mu 최대 마찰, Fz 축 하중."""
    z = np.tan(alpha)
    F = C * z - C ** 2 / (3 * mu * Fz) * np.abs(z) * z + C ** 3 / (27 * mu ** 2 * Fz ** 2) * z ** 3
    return np.where(np.abs(z) < 3 * mu * Fz / C, F, mu * Fz * np.sign(z))


def est_fiala(names, m, a, b, Iz, eta, gears, wheel_r):
    """축별 옆 힘(단일 트랙 식)과 미끄럼각에 Fiala 곡선을 맞춤 → 강성 C, 최대 마찰 μ (앞, 뒤).
    축 힘: Fyr = (a·m·ay − Iz·ṙ)/L, Fyf·cosδ = (b·m·ay + Iz·ṙ)/L − Fxf·sinδ. 표본: 8 m/s 이상, |Fxf| < 1 kN (결합 미끄럼 배제),
    |ṙ| < 1 rad/s² (요 관성 영향 축소). 5 Hz 평활, 100 Hz. 같은 방법이 GTI 데이터에서 공개값(Spielberg 127/171)을 4~5% 로 재현 [11/21]."""
    from scipy.optimize import least_squares
    L = a + b
    col = dict(ux="Vx_mps", uy="Vy_mps", r="yawRate_radps", delta="roadWheelAngle_rad", fxf="Fxf")
    V = {k: [] for k in col}
    for n in names:
        df = D.read_raw(n, gears=gears, wheel_r=wheel_r, drive_eff=eta)
        for k, c in col.items():
            V[k].append(S.smooth(df[c].to_numpy(float), 5.0, 200)[400:-400:2])
    V = {k: np.concatenate(v) for k, v in V.items()}
    vyd, rd = S.deriv(V["uy"], 100), S.deriv(V["r"], 100)
    ay = vyd + V["ux"] * V["r"]
    Fyr = (a * m * ay - Iz * rd) / L
    Fyf = ((b * m * ay + Iz * rd) / L - V["fxf"] * np.sin(V["delta"])) / np.cos(V["delta"])
    af, ar = S._alphas(a, L, V["ux"], V["uy"], V["r"], V["delta"])
    k = (V["ux"] > 8.0) & (np.abs(rd) < 1.0) & (np.abs(V["fxf"]) < 1000.0)
    out = {}
    for ax_, al, F, Fz in (("front", af[k], Fyf[k], m * G * b / L), ("rear", ar[k], Fyr[k], m * G * a / L)):
        sol = least_squares(lambda p: (fiala(al, p[0] * 1e5, p[1], Fz) - F) / Fz, [1.2, 1.0], bounds=([0.1, 0.3], [5.0, 2.0]))
        out[ax_] = dict(C=float(sol.x[0] * 1e5), mu=float(sol.x[1]))
    out["n"] = int(k.sum())
    return out


def build_specs():
    """specs.json 내용. 이전 specs.json 을 읽지 않는다 (순서: 반지름·감속비 → 구동력 → 효율 → Fiala·상한표)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import drive_force_limit as DFL
    import fit_delays as FD
    names = D.available()
    nf = len(names)
    it = {}

    def put(key, value, source, basis, **extra):
        it[key] = dict(value=value, source=source, basis=basis, **extra)
        print(f"{key}: {value if not isinstance(value, dict) else '(표)'}  [{source}]", flush=True)

    # ---- 공개값 (데이터 차량 = Stanford DDL 'Niki' 2018 VW Golf VII GTI 가정 [추정], 계획서 [결정 11/21] 2)
    NIKI = "데이터 차량 = Niki 가정 [추정, 이름으로 확인된 자료 없음]"
    put("mass", 1777.0, "published", f"Niki 1,778 kg (Spielberg 박사논문 2021 표 4.3), 썬더힐 GTI 1,776 kg (Subosits 박사논문 2020 부록 A)의 평균. "
        f"{NIKI}. 데이터는 '힘/질량' 비율만 정해 질량을 식별하지 못함(기준 A). 불확실 +5% (자매차 Norbert 2024 1,868 kg)", uncertainty=[1777.0, 1868.0])
    put("cg_a", 1.194, "published", f"무게중심→앞축. Spielberg 2021·Stanford 코드·Weber 2024 모두 1.194 m (Subosits 1.19). {NIKI}. "
        "데이터 회귀는 GTI −19.6%, AV-21 −52.2% 로 식별 안 됨(기준 A). 이전 가정 앞 무게 몫 0.61 → 1.026 m")
    put("cg_b", 1.436, "published", "무게중심→뒤축. Spielberg 2021 1.436 m (Stanford 코드 1.437, Subosits 1.44)")
    put("track_f", 1.54, "published", "앞 윤거. Subosits 2020 부록 A (이전 값 1.541 = Golf Mk7 공개 제원 가운데)")
    put("track_r", 1.52, "published", "뒤 윤거. Subosits 2020 부록 A (이전 1.513)")
    put("cg_height", 0.56, "published", "무게중심 높이. Subosits 2020 0.56 m (Weber 2024 Norbert 0.55, Stanford 코드 0.55). 이전에는 Escort 값을 빌림")
    put("yaw_inertia", 2763.49, "published", "Niki: Spielberg 2021·Stanford 코드 일치. 데이터 회귀는 정방향/역방향이 GTI 1,094~3,529 (정답 2,763) 로 식별 안 됨 [11/21]. "
        "공개값끼리도 다름(Subosits 3,950, Norbert 3,049) → 기준 A·B 모두 아니오인 경우: Niki 값 + 불확실 범위 기록 (사용자 승인 11/21)", uncertainty=[2763.49, 3950.0])
    # ---- 가정값
    put("roll_gradient_target", 0.30, "assumed", "도/(m/s²). 현가 센서 추정 0.23 (단위 cm 가정)과 일반 승용차 0.3~0.6 이 겹치는 값 (11/19)")
    put("roll_front_share", 0.61, "assumed", "롤 강성 앞 몫. Escort 원래 값(스프링+안티롤바) 유지 (11/19)")
    put("abs_slip", [0.12, 0.18], "assumed", "이 미끄럼률 사이에서 브레이크 토크 1→0. 타이어 최대 마찰은 미끄럼 약 0.15 (11/18). 데이터 제동 시간 1% 에서 ABS 작동")
    m = it["mass"]["value"]
    put("resist_cr0", CR0_COEF * m * G, "assumed", f"굴림 저항 (N) = {CR0_COEF}·m·g (일반 승용차 굴림 계수)")
    put("resist_cr2", CR2_ASSUMED, "assumed", "공기 저항 계수 (N/(m/s)²) = 0.5·1.2·Cd 0.30·A 2.2 (일반 승용차)")
    # ---- 데이터 (20개 파일)
    wr, wi = est_wheel_radius(names)
    put("wheel_radius", wr, "data", "구동되지 않는 뒷바퀴: 차속 ÷ 회전 속도, 직진(|r|<0.05)·뒤 제동 없음·5 m/s 이상 표본의 중앙값. 이전 0.319 (2개 파일)", files=nf, **wi)
    gr, gi = est_gear_ratios(names)
    put("gear_ratios", gr, "data", "엔진 회전 ÷ 앞바퀴 평균 회전 분포의 봉우리 4곳 (±3% 안 중앙값). 이전 [8.8, 5.65, 3.93, 3.0] (2개 파일). "
        "데이터의 GearRatio 열은 실측의 약 2.41배라 쓰지 않음", files=nf, **gi)
    eta, ei = est_eta(names, m, it["resist_cr0"]["value"], CR2_ASSUMED, gr, wr)
    put("drive_eff", eta, "data", "직진 구동 구간, 질량 공개값 고정, OSM 경사 포함, 힘 100 ms 지연 보정. 바퀴·구동계 회전 관성을 흡수한 '실효' 효율 "
        "(시뮬레이터는 바퀴 관성을 따로 계산하므로 sim.accel_input 에서 구동 입력을 보정, 11/21 결정 (가)). 이전 0.699 (질량 1,497·경사 없음)", files=nf, **ei)
    share, bmin, bi = est_brake(names)
    put("brake_front_share", share, "data", "속도 > 5 m/s · 네 바퀴 제동력 합 ≥ 2 kN 표본의 앞 몫 중앙값. 주행 중 자동 운전 제어기가 뒤 압력을 앞의 약 0.64배로 명령한 결과 "
        "(정지 중 같은 압력이면 90.9/(90.9+47.2)=0.658 = 공개 정적 배분 65:35 와 일치). 이 데이터 재생에는 주행 중 값이 맞음", files=nf,
        n=bi["share_n"], per_file_min_max=bi["share_per_file_min_max"])
    put("brake_force_min", bmin, "data", "앞바퀴 힘 하한 (N). 5 m/s 이상·앞 제동력 > 500 N 표본의 상위 1%. 이전 −6 kN (2개 파일 하위 1%)", files=nf, n=bi["front_n"])
    sr, si = est_steer_rate(names)
    put("steer_rate_max", sr, "data", "자동 조향·5 m/s 이상·10 Hz 저역 통과 뒤 조향 속도 99.9%. 장비 상한(Niki 90°/s = 1.57 rad/s)이 아니라 데이터 범위를 쓴다: "
        "모델이 학습한 조작 범위 안에서 비교하기 위함 (사용자 결정 11/21, 계획서 [결정 11/21] 3). 최대값은 센서 튐으로 봄", files=nf, **si)
    ayt, ai = est_ay_target(names)
    put("ay_target", ayt, "data", "마찰 보정 목표 (m/s²): 실제 옆 가속도 dvy/dt + vx·r 상위 1%. vx·r 은 옆 미끄럼 속도 변화를 빼먹어 크게 나옴 (11.66 → 10.66, 11/21 사용자 승인). "
        "필터는 모델 학습 데이터와 같은 10 Hz (사용자 결정 11/21: 학습과 다르게 할 근거 없음; 2 Hz 면 10.48). "
        "교차 확인: 관성 센서 ayCG 와 상관 0.994·평균 차 0 (10 Hz 상위 1% 는 센서 진동으로 11.03, 2 Hz 에서 10.57 vs 10.48 로 일치), "
        "Fiala 축별 마찰 1.06/1.12", files=nf, **ai)
    fi = est_fiala(names, m, it["cg_a"]["value"], it["cg_b"]["value"], it["yaw_inertia"]["value"], eta, gr, wr)
    fb = "Fiala(브러시) 곡선을 축별 옆 힘·미끄럼각에 맞춤 (연구실 논문과 같은 방법, GTI 데이터에서 Spielberg 127/171 kN/rad 를 4~5% 로 재현). 이전 85.7/75.0 (직선, 회귀 희석으로 작게 나옴)"
    put("tire_stiffness_f", fi["front"]["C"], "data", "앞 축 코너링 강성 (N/rad). " + fb, files=nf, n=fi["n"])
    put("tire_stiffness_r", fi["rear"]["C"], "data", "뒤 축 코너링 강성 (N/rad). " + fb, files=nf, n=fi["n"])
    put("tire_mu_f", fi["front"]["mu"], "data", "앞 축 최대 마찰 (Fiala). 시뮬레이터 마찰 보정의 교차 확인값 (직접 쓰지 않음), 모델 타이어 초기값 D 에 씀", files=nf)
    put("tire_mu_r", fi["rear"]["mu"], "data", "뒤 축 최대 마찰 (Fiala). 위와 같음", files=nf)
    # 구동력 상한표: 200 Hz 전 표본 (drive_force_limit.table)
    v, F = [], []
    for n in names:
        df = D.read_raw(n, gears=gr, wheel_r=wr, drive_eff=eta)
        v.append(df["Vx_mps"].to_numpy()); F.append(eta * df["F_drive"].to_numpy())
    put("drive_force_cap", DFL.table(np.concatenate(v), np.concatenate(F), m), "data",
        "속도 2 m/s 구간별 구동력(효율 포함) 최댓값·상위 1%. 시뮬레이터·MPPI 의 구동 상한 (32 m/s 이상은 sim 이 일정 출력으로 이음)", files=nf)
    dl = FD.estimate()
    put("actuator_delays", {k: dict(use=dl[k]["use"], fit=dl[k]["fit"], rms=dl[k]["rms"], rms_no_delay=dl[k]["rms_no_delay"]) for k in ("brake", "engine")},
        "data", "자동 운전 구간의 명령 → 실제 (브레이크 압력, 엔진 토크): 순수 지연 + 1차 지연 (초). 지연 없음보다 오차가 1% 이상 줄 때만 사용 (use)", files=nf)
    return it


def main():
    import datetime
    import subprocess
    from deep_dynamics.golf import golf_specs
    items = build_specs()
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    doc = dict(note="Golf 차량 계수 (estimate_specs.py 가 만듦, 손으로 고치지 말 것). 출처 규칙·결정: docs/experiment_plan_golf.md [결정 11/21]",
               generated=datetime.datetime.now().isoformat(timespec="seconds"), code_commit=commit, data_files=D.available(), items=items)
    json.dump(doc, open(golf_specs.SPECS_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    golf_specs.load.cache_clear()
    os.makedirs(OUT, exist_ok=True)
    L = ["# Golf 차량 계수 (deep_dynamics/golf/specs.json)", "", f"생성 {doc['generated']}, 코드 {commit}, 데이터 {len(doc['data_files'])}개 파일", "",
         "| 항목 | 값 | 출처 | 근거 |", "|---|---|---|---|"]
    for k, v in items.items():
        val = v["value"] if not isinstance(v["value"], dict) else "(표, json 참고)"
        L.append(f"| {k} | {val} | {v['source']} | {v['basis']} |")
    open(os.path.join(OUT, "specs_build.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"→ {golf_specs.SPECS_PATH}")
    validation_report()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recheck":
        recheck()
    else:
        main()
