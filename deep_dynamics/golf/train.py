"""사다리 모델 학습 (데이터와 무관한 부분만). 지금까지 실험의 src/iac_train.py, src/rv_train.py 와 같은 방식.

  - 손실: 상태별 1스텝 변화량 표준편차로 나눈 MSE (vx·vy·r 단위 차이 보정)
  - roll=1  → 논문식 1스텝 학습
    roll=K  → 기록된 입력으로 K 스텝 굴리며 매 스텝 오차 (k 번째 스텝 오차는 1/k 가중). 우리 실험의 "1초 손실"은 K=25 (dt 0.04)
  - Adam, 전역 물리 계수(GlobalPhys.raw, AxleNet.logf)는 학습률 3배, 기울기 노름 10 자름
  - [변경 11/18] 모든 모델·손실에 같은 규칙 (docs/experiment_plan_golf.md 4-2절):
      학습률: 검증 손실이 PLATEAU_PATIENCE 에폭 동안 상대 IMPROVE_TOL 이상 안 나아지면 절반 (ReduceLROnPlateau)
      조기 종료: 검증 손실이 STOP_PATIENCE 에폭 동안 안 나아지면 멈춤. 최대 MAX_EPOCHS
      검증 손실이 가장 낮았던 에폭의 모델을 돌려줌
    (이전: 코사인 감소 + 정해진 에폭 + 마지막 에폭 모델)

입력 배열 모양 (windows() 로 만듦)
  X (N, H, 6) 이력 창, Y (N, K, 3) 다음 K 스텝 정답 상태, U (N, K-1, 3) 2번째 스텝부터의 입력
"""
import time

import numpy as np
import torch

from .ladder import NF, Norm, build


def windows(segments, H, horizon=1, stride=1):
    """segments: [(s (T,3), u (T,3)), ...] 끊김 없는 구간 목록 → (X, Y, U).
    창은 구간 경계를 넘지 않음. 학습·검증·시험은 반드시 '주행(파일) 단위'로 먼저 나눈 뒤 각각 부를 것
    (창을 섞은 뒤 나누면 겹치는 창이 양쪽에 들어가 누수)."""
    X, Y, U = [], [], []
    for s, u in segments:
        su = np.concatenate([s, u], 1)
        for k in range(H - 1, len(su) - horizon, stride):
            X.append(su[k - H + 1:k + 1])
            Y.append(s[k + 1:k + 1 + horizon])
            U.append(u[k + 1:k + horizon])
    X = np.asarray(X, dtype=np.float32).reshape(-1, H, NF)
    Y = np.asarray(Y, dtype=np.float32).reshape(-1, horizon, 3)
    U = np.asarray(U, dtype=np.float32).reshape(len(X), horizon - 1, 3)
    return X, Y, U


def loss_fn(model, x, y, u, w):
    roll = y.shape[1]
    hist, tot = x, 0.0
    for k in range(roll):
        s = model(hist)
        tot = tot + ((((s - y[:, k]) * w) ** 2) / (k + 1)).mean()
        if k < roll - 1:
            hist = torch.cat([hist[:, 1:], torch.cat([s, u[:, k]], 1).unsqueeze(1)], 1)
    return tot / roll


MAX_EPOCHS = 300
PLATEAU_PATIENCE = 10
STOP_PATIENCE = 25
IMPROVE_TOL = 1e-3  # 상대 개선 기준


def fit(rung, cfg, train, val, max_epochs=MAX_EPOCHS, lr=1e-3, bs=1024, seed=0, device="cpu", log_every=25):
    """train, val: windows() 결과 (X, Y, U). 표준화 상수는 train 에서만 계산.
    반환 (검증 최저 모델, 기록). 기록 hist 의 행 = (에폭, 학습 손실, 검증 손실, 그 에폭의 기본 학습률)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Utr = train
    norm = Norm(Xtr, Ytr)
    model = build(rung, cfg, norm, device)
    T = lambda z: torch.as_tensor(z, dtype=torch.float32, device=device)
    Xtr_t, Ytr_t, Utr_t = T(Xtr), T(Ytr), T(Utr)
    Xva_t, Yva_t, Uva_t = (T(z) for z in val)
    w = 1.0 / norm.dsd.to(device)

    is_fast = lambda n: n.endswith("g.raw") or n.endswith("logf")
    fast = [p for n, p in model.named_parameters() if is_fast(n)]
    rest = [p for n, p in model.named_parameters() if not is_fast(n)]
    groups = [dict(params=rest, lr=lr)] + ([dict(params=fast, lr=3 * lr)] if fast else [])
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=PLATEAU_PATIENCE,
                                                       threshold=IMPROVE_TOL, threshold_mode="rel")

    t0 = time.time()
    hist, best, best_state = [], float("inf"), None
    best_epoch = ref_epoch = -1  # 검증 최저 에폭 / 스케줄러가 '의미 있는 개선'으로 본 마지막 에폭
    stop_reason = "max_epochs"
    for ep in range(max_epochs):
        cur_lr = opt.param_groups[0]["lr"]
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        tot, used = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model, Xtr_t[idx], Ytr_t[idx], Utr_t[idx], w)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += loss.item() * len(idx)
            used += len(idx)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(Xva_t), 4096):
                sl = slice(i, i + 4096)
                vl += loss_fn(model, Xva_t[sl], Yva_t[sl], Uva_t[sl], w).item() * len(Xva_t[sl])
            vl /= max(len(Xva_t), 1)
        tl = tot / used if used else float("nan")  # 손실이 유한하지 않아 건너뛴 배치는 평균에서 뺌
        hist.append((ep, tl, vl, cur_lr))
        # 조기 종료는 스케줄러 자신의 상태(sched.best: 마지막 '의미 있는' 개선 때의 값)로 판단한다.
        # 별도 기준(예: 최솟값)으로 세면 스케줄러와 어긋나 학습률이 줄기 전에 멈출 수 있음 (11/18 테스트에서 확인)
        prev = sched.best
        sched.step(vl)
        if sched.best != prev:
            ref_epoch = ep
        if vl < best:  # 돌려줄 모델은 검증 손실 최솟값 기준
            best, best_epoch = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log_every and (ep % log_every == 0 or ep == max_epochs - 1):
            print(f"{rung} ep {ep} train {tl:.4f} val {vl:.4f} best {best:.4f}@{best_epoch} lr {cur_lr:.2e} {time.time() - t0:.0f}s", flush=True)
        if ep - ref_epoch >= STOP_PATIENCE:
            stop_reason = "no_improvement"
            break
    if best_state is None:  # 검증 손실이 한 번도 유한하지 않았음 → 마지막 모델 그대로 (기록으로 알 수 있음)
        stop_reason += "_no_finite_val"
    else:
        model.load_state_dict(best_state)
    print(f"{rung} stop ep {ep} ({stop_reason}), best val {best:.4f}@{best_epoch}", flush=True)
    return model.eval(), dict(hist=hist, best_val=best, best_epoch=best_epoch, ref_epoch=ref_epoch, stop_epoch=ep,
                              stop_reason=stop_reason, sec=time.time() - t0)
