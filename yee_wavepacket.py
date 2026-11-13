# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# 실행: python yee_wavepacket.py
# 수치 계산·그래프와 보고서 4.6의 오차 원인 분석을 함께 실행한다.
# 계산 결과는 wavepacket_results, 오차 분석은 error_validation에 저장한다.
# 선택: python yee_wavepacket.py --output-dir 원하는 폴더
# 검증만 실행: python yee_wavepacket.py --causal-only
# 검증 저장 위치: python yee_wavepacket.py --validation-dir 원하는 폴더
#
# 수치해는 아래 simulate()에서 Maxwell 방정식을 직접 적분한다.
# 파동방정식이나 해석해를 수치장의 갱신에 사용하지 않는다.
#
# 읽는 순서: make_grid -> simulate -> measurements -> causal_analysis -> main.
# 계산 변수는 tau=c*t [m], b=c*B_y [V/m]이다. E_x와 b의 크기를 비슷하게
# 맞춰 계산하고, B_y를 그리거나 저장할 때 b를 c로 나누어 T 단위로 바꾼다.
# 출력 그림은 과학적 표기 계수를 소수 둘째 자리까지 표시한다.
# 계산 배열과 JSON/CSV/NPZ에는 반올림 전 값을 저장한다.
#
# 참고:
# Griffiths, Introduction to Electrodynamics, 5판(2024),
# §§7.3.3, 7.3.5-7.3.6, 9.2.2, 9.3.1-9.3.2.
# J. B. Schneider, Understanding the FDTD Method, Ch. 3,
# https://eecs.wsu.edu/~schneidj/ufdtd/chap3.pdf

'''
함께 토의한 사람 없음
사용한 AI: GPT-6.1 Sol 
입력한 질문:
1. Yee-type staggered grid에 대한 정보가 필요해, 관련 정보를 얻을 수 있는 교재나 사이트를 제안해줘. 그리고 이게 뭔지 간략히 설명하고, 이를 어떻게 구현하는지 제시해줘.
2. Leapfrog 방법이 정확히 어떻게 구현되고 왜 이 방법을 사용하는지 알려줘. 
3. 시공간을 반 칸씩 어긋나게 놓는 시간 적분 방식이 Leapfrog 방법이라는 건 알겠는데, 이게 어떤 장점이 있는지 구체적으로 설명해줘.
4. 시간 간격과 공간 간격이 구체적으로 어떤 조건에서 안정적이라고 할 수 있는거지?
5. 그래서 Yee 격자에서 전기장과 자기장의 위치와 시각을 반 칸씩 어긋나게 두는 이유가 뭐야?
6. 문제 3에서 파묶음이 $+z$ 방향으로만 진행하도록 정한 초기 자기장은 왜 그렇게 추정할 수 있나?
7. 전기장 진폭과 자기장 진폭으로 각각 반사율·투과율을 계산하는 식을 함께 제시해줘.
8. 가능한 오차 원인을 나열하는 데 그치지 않고, 실제 원인을 분리해 검증할 실험 계획을 세워줘. 그 실험을 실행하여 주된 오차 원인이 무엇인지 분석하고 검증 결과를 설명해줘.
9. simulate()에서 자기장을 먼저 계산하고 전기장을 나중에 계산하는 이유가 뭐야? 이 순서로 해야 하는 이유를 설명해줘.
10. 반복문에서 사용하는 전기장과 자기장은 각각 어느 시각의 값인 거야? 갱신 전후로 구분해서 설명해줘.
11. np.diff(e)랑 np.diff(b_plus)가 각각 어떤 미분을 계산하는 거야? 갱신식에 마이너스가 붙는 이유도 같이 설명해줘.
12. 전기장 배열이 자기장 배열보다 하나 더 긴데, 왜 이렇게 되는 거야?
13. 자기장을 그냥 B_y로 계산하지 않고 b=cB_y로 바꿔서 계산하는 이유가 있어? 그래프에서는 다시 어떻게 바꾸는 거야?
14. b_minus = b_zero + 0.5 * COURANT * np.diff(e)는 어떻게 나온 식이야? 초기 자기장을 그대로 넣으면 안 되는 건가?
15. 그래프를 그릴 때 자기장은 왜 b_minus랑 b_plus를 평균해서 사용하는 거야?
16. material_fraction이 경계에서 유전체가 차지하는 비율을 어떻게 계산하는 건지 설명해줘.
17. 출력 시각을 반복 횟수로 바꾸면서 반올림했는데, 왜 그다음에 시간 격자와 일치하는지 다시 확인하는 거야?
18. 진폭을 구할 때 그냥 배열의 최댓값을 사용하면 안 돼? 세 점으로 보간하는 이유가 뭐야?
19. 입사파, 반사파, 투과파의 진폭은 각각 어느 구간에서 구하는 거야? 반사 전기장은 음수인데 이 부호는 어떻게 처리하고 있어?
20. 그래프 코드만 수정했을 때 계산 결과는 그대로라는 걸 어떻게 확인할 수 있어? 실제로 비교해서 확인하는 방법을 설명해줘. 
'''

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

# 화면이나 GUI가 없는 터미널에서도 동작하도록 파일 출력용 backend를 사용한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent


# eps0*mu0*c^2=1이 되도록 상수를 정의한다. 이 문제의 무차원 R, T는
# 두 매질의 상대 물성으로 결정된다. 아래 상수는 반올림하지 않고 계산한다.
C = 299_792_458.0             # 진공 광속 [m/s]
MU0 = 4.0e-7 * np.pi          # [H/m], 교재 수준의 근삿값
EPS0 = 1.0 / (MU0 * C * C)    # [F/m]
SIGMA = 1.0                  # Gaussian 폭 [m]
E0 = 1.0                     # 초기 전기장 최댓값 [V/m]
INTERFACE = 5.0              # 진공/유전체 경계 [m]
EPSR2 = 4.0                  # 유전체의 상대 유전율 (진공은 1)
N2 = np.sqrt(EPSR2)           # mu_r=1인 비자성, 무손실, 비분산 매질 가정
ZMIN, ZMAX = -20.0, 20.0      # 계산 영역 [m]; 그림은 파묶음 주변만 확대한다.
COURANT = 0.5                # S=c*dt/dz <= 1 (진공에서 CFL 조건)
MAIN_DZ = 0.01               # [m]


def gaussian(z):
    """초기 전기장 E0*exp(-z^2/sigma^2) [V/m]. z는 스칼라 또는 배열이다."""
    return E0 * np.exp(-(np.asarray(z) / SIGMA) ** 2)


def make_grid(dz, dielectric):
    """E 격자점, 반 칸 옮긴 B 격자점, E 주변 셀의 평균 상대 유전율을 반환한다.

    dz는 격자 간격 [m], dielectric은 문제 4의 유전체 사용 여부이다.
    E 배열은 양 끝점을 포함하므로 B 배열보다 원소가 하나 많다.
    """
    intervals = int(round((ZMAX - ZMIN) / dz))
    if not np.isclose(intervals * dz, ZMAX - ZMIN):
        raise ValueError("dz must divide the domain length.")
    z_e = ZMIN + dz * np.arange(intervals + 1)
    z_b = 0.5 * (z_e[:-1] + z_e[1:])
    if dielectric:
        # 각 E 격자점의 주변 셀 [z_j-dz/2, z_j+dz/2]에 Ampere 법칙을
        # 적분하고, E_j를 셀의 대표 전기장으로 두어 유전율을 공간 평균한다.
        # material_fraction은 셀에서 유전체가 차지하는 길이의 비율이다.
        # 과제에 사용한 격자에서는 경계점의 셀에 진공과 유전체가 반씩
        # 포함되어 평균 상대 유전율이 (1+4)/2=2.5가 된다.
        material_fraction = np.clip((z_e + dz / 2 - INTERFACE) / dz, 0, 1)
        eps_r = 1.0 + (EPSR2 - 1.0) * material_fraction
    else:
        eps_r = np.ones_like(z_e)
    return z_e, z_b, eps_r


def simulate(dz=MAIN_DZ, dielectric=False, right_only=False,
             snapshot_tau=(0.0, 2.0)):
    """Maxwell 방정식을 Leapfrog 방법으로 시간 적분한다.

    dz: 공간 간격 [m]. dielectric: 문제 4의 매질 사용 여부.
    right_only: 문제 3의 진공 초기장 b(0)=E(0) 사용 여부.
    snapshot_tau: 저장할 시각들의 tau=c*t 값 [m]; 시간 격자에 맞아야 한다.

    계산하는 두 1차 방정식은 다음과 같다.
       db/dtau = -dE/dz,     eps_r*dE/dtau = -db/dz.
    step=n의 반복 시작에서 e는 E^n, b_minus는 b^(n-1/2)이다.
    b_plus=b^(n+1/2)를 구한 뒤 E^(n+1)를 구하고 다음 반복으로 간다.

    반환 딕셔너리에는 격자, 간격, 유전율과 snapshots가 들어간다.
    snapshots[tau]의 E와 b는 같은 시각의 복사본이며 b의 단위는 V/m이다.
    energy와 invariant의 단위는 단위 횡단면적당 에너지 [J/m^2]이다.
    """
    z_e, z_b, eps_r = make_grid(dz, dielectric)
    d_tau = COURANT * dz
    dt = d_tau / C
    # 국소 파동 속도 v=c/sqrt(eps_r)에 대해 v*dt/dz<=1을 확인한다.
    # 이 과제에서는 진공의 속도가 가장 크므로 진공에서 조건이 가장 엄격하다.
    if COURANT > np.sqrt(np.min(eps_r)):
        raise ValueError("CFL stability condition violated.")
    wanted = {int(round(tau / d_tau)): float(tau) for tau in snapshot_tau}
    for step, tau in wanted.items():
        if not np.isclose(step * d_tau, tau, rtol=0, atol=1e-12):
            raise ValueError("Requested snapshot does not lie on the time grid.")

    e = gaussian(z_e)
    e[[0, -1]] = 0.0
    # b=cB이므로 진공에서 B(0)=E(0)/c인 문제 3은 b(0)=E(0)로 설정한다.
    b_zero = gaussian(z_b) if right_only else np.zeros_like(z_b)

    # 주어진 초기 B는 t=0의 값이다. 이것을 그대로 t=-dt/2에 넣으면
    # 시간 오차가 생긴다. Faraday 법칙으로 반 step을 뒤로 이동한다.
    # b^{-1/2}=b(0)+(S/2)*(E_{j+1}^0-E_j^0).
    # Taylor 전개에서 1차 시간 미분항까지 반영한 2차 정확도의 초기화다.
    # 첫 b_plus와 b_minus의 평균은 지정한 b(0)를 재현한다.
    b_minus = b_zero + 0.5 * COURANT * np.diff(e)
    snapshots = {}
    for step in range(max(wanted) + 1):
        # Faraday 법칙: B의 y성분. 부호는 (curl E)_y=+dE_x/dz에서 나온다.
        b_plus = b_minus - COURANT * np.diff(e)
        if step in wanted:
            # E^n과 같은 시각의 B가 필요하므로 양쪽 시간값을 평균한다.
            # 공간위치는 그대로 z_{j+1/2}; 그래프에도 그 위치를 사용한다.
            b_sync = 0.5 * (b_minus + b_plus)
            # 물리적 에너지 밀도는 eps0*(eps_r*E^2+b^2)/2이다.
            # dz를 곱해 z 방향으로 적분하면 단위 횡단면적당 에너지가 된다.
            energy = 0.5 * EPS0 * dz * (
                np.sum(eps_r * e**2) + np.sum(b_sync**2))
            # 이산 보존량에는 동시각 b^2 대신 b_minus*b_plus가 들어간다.
            # 시간에 무관한 eps_r와 E=0 끝점 조건에서 보존되는 양이며,
            # 동시각 에너지와는 고정된 공간 격자에서 O(dt^2)만큼 다르다.
            invariant = 0.5 * EPS0 * dz * (
                np.sum(eps_r * e**2) + np.sum(b_minus * b_plus))
            snapshots[wanted[step]] = {
                "E": e.copy(), "b": b_sync.copy(),
                "energy": float(energy), "invariant": float(invariant)}
        if step == max(wanted):
            break
        # Ampere-Maxwell 법칙: (curl H)_x=-dH_y/dz.
        # mu=mu0 일정이므로 eps_r를 나누어 E를 갱신한다.
        e[1:-1] -= COURANT / eps_r[1:-1] * np.diff(b_plus)
        # 양 끝은 PEC에 해당하는 E=0 조건이다. ct<=10 m 동안 주요 파묶음은
        # 끝점에 도달하지 않는다. Gaussian 꼬리는 무한히 퍼져 있지만
        # 끝점에 도달하는 꼬리의 크기는 약 E0*exp(-100)/2로 매우 작다.
        e[[0, -1]] = 0.0
        b_minus = b_plus
    return {"z_e": z_e, "z_b": z_b, "eps_r": eps_r, "dz": dz,
            "dt": dt, "d_tau": d_tau, "snapshots": snapshots}






def peak(z, values, lo, hi, sign=1):
    """등간격 격자에서 세 점의 포물선 보간으로 극값 진폭과 위치를 추정한다.

    [lo, hi]에서 sign=1은 최댓값, sign=-1은 최솟값을 찾는다.
    선택한 점의 양옆 원소가 있어야 한다. sign을 되돌려 부호 있는 진폭을
    반환하므로, 반사 전기장 진폭은 음수로 유지된다. 보간은 격자 사이에
    놓인 극값의 추정 오차를 줄인다. 수치장의 차분 오차는 별도로 남는다.
    """
    candidates = np.flatnonzero((z >= lo) & (z <= hi))
    index = int(candidates[np.argmax(sign * values[candidates])])
    y_l, y_c, y_r = sign * values[index - 1:index + 2]
    denominator = y_l - 2 * y_c + y_r
    delta = 0.0 if denominator == 0 else 0.5 * (y_l - y_r) / denominator
    delta = float(np.clip(delta, -1.0, 1.0))
    fitted = y_c - 0.25 * (y_l - y_r) * delta
    return {"amplitude": float(sign * fitted),
            "position": float(z[index] + delta * (z[1] - z[0]))}


def measurements(vacuum, medium):
    """문제 4의 입사·반사·투과 진폭, R/T 및 에너지를 구한다.

    vacuum은 tau=3, medium은 tau=0과 10의 저장값이 필요하다.
    두 결과의 dz는 같아야 한다. 반환 Bi_T/Br_T/Bt_T의 단위는 T이다.
    """
    # 입사파는 같은 격자의 진공 기준 계산 c*t=3 m에서 측정한다. 반대쪽
    # 초기 파와 충분히 떨어져 있고 진폭은 약 0.5 V/m이다. E0=1 V/m를
    # 입사 진폭으로 사용하면 안 된다. B0=0인 초기장은 두 진행파의 합이다.
    inc = vacuum["snapshots"][3.0]
    final = medium["snapshots"][10.0]
    ie = peak(vacuum["z_e"], inc["E"], 2, 4)
    ib = peak(vacuum["z_b"], inc["b"], 2, 4)
    re = peak(medium["z_e"], final["E"], -2, 2, sign=-1)
    # -z로 가는 반사파에서는 b=-n1*E이다. 따라서 반사 E는 음수,
    # 반사 b는 양수인 극값을 각각 측정한다. 원래의 왼쪽 파는 z=-10에 있다.
    rb = peak(medium["z_b"], final["b"], -2, 2)
    te = peak(medium["z_e"], final["E"], 6, 9)
    tb = peak(medium["z_b"], final["b"], 6, 9)
    ai, ar, at = (p["amplitude"] for p in (ie, re, te))
    bi, br, bt = (p["amplitude"] for p in (ib, rb, tb))
    # B는 b=cB로 저장되어 있지만 진폭 비에서는 c가 약분된다.
    # E 기준 T=(n2/n1)*(At/Ai)^2. B 기준 T=(v2/v1)*(Bt/Bi)^2.
    # S_z=E*B/mu0이고 진행파 B=+/-nE/c이므로 두 식의 굴절률 인자가 다르다.
    # R, T의 분모는 경계로 들어가는 오른쪽 입사파의 세기이다.
    metrics = {
        "dz": medium["dz"], "Ai": ai, "Ar": ar, "At": at,
        "Bi_T": bi / C, "Br_T": br / C, "Bt_T": bt / C,
        "r_E": ar / ai, "t_E": at / ai,
        "R_E": (ar / ai)**2, "T_E": N2 * (at / ai)**2,
        "R_B": (br / bi)**2, "T_B": (bt / bi)**2 / N2,
        "z_reflected": re["position"], "z_transmitted": te["position"],
    }
    metrics["R_plus_T"] = metrics["R_E"] + metrics["T_E"]
    initial = medium["snapshots"][0.0]
    metrics["physical_energy_ratio"] = final["energy"] / initial["energy"]
    metrics["invariant_relative_drift"] = max(
        abs(s["invariant"] / initial["invariant"] - 1)
        for s in medium["snapshots"].values())
    # 최종 시각에는 원래 왼쪽 파, 반사파, 투과파가 분리되어 있다.
    # 공간 구간을 나눠 적분한 펄스 에너지로 진폭 기반 R, T를 교차 확인한다.
    # 각 적분값은 단위 횡단면적당 에너지 [J/m^2]이다.
    def region_energy(lo, hi):
        em = (medium["z_e"] >= lo) & (medium["z_e"] < hi)
        bm = (medium["z_b"] >= lo) & (medium["z_b"] < hi)
        return float(0.5 * EPS0 * medium["dz"] * (
            np.sum(medium["eps_r"][em] * final["E"][em]**2)
            + np.sum(final["b"][bm]**2)))
    # 초기 B=0인 파묶음은 같은 에너지를 갖는 좌우 성분으로 분리된다.
    # 이 에너지 정규화는 초기 유전체 꼬리의 극히 작은 기여를 무시한다.
    u_inc = initial["energy"] / 2.0
    metrics["R_energy"] = region_energy(-5, 5) / u_inc
    metrics["T_energy"] = region_energy(5, ZMAX + 1) / u_inc
    metrics["original_left_energy_fraction"] = (
        region_energy(ZMIN, -5) / initial["energy"])
    return metrics




def peak_amplitude(z, y, lo, hi, sign=1, degree=2):
    """오차 분석용 진폭을 반환한다. 기본값은 공통 세 점 보간을 사용한다."""
    if degree == 2:
        return peak(z, y, lo, hi, sign)["amplitude"]
    mask = np.flatnonzero((z >= lo) & (z <= hi))
    j = mask[np.argmax(sign * y[mask])]
    radius = degree // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    coeff = np.polynomial.polynomial.polyfit(x, sign * y[j-radius:j+radius+1], degree)
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coeff))
    choices = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and abs(r.real) <= 1]
    assert choices
    return sign * max(np.polynomial.polynomial.polyval(q, coeff) for q in choices)


def measure_error_coefficients(vac, med, time=10., degree=2, exact_incident=False):
    """입사·반사·투과 진폭에서 계수와 이론값에 대한 부호 있는 오차를 구한다."""
    vi, mf = vac["snapshots"][3.], med["snapshots"][time]
    cr, ct = 10-time, 5+(time-5)/2
    args = [(vac["z_e"], vi["E"], 2, 4, 1),
            (med["z_e"], mf["E"], cr-1.8, cr+1.8, -1),
            (med["z_e"], mf["E"], ct-1.1, ct+1.1, 1),
            (vac["z_b"], vi["b"], 2, 4, 1),
            (med["z_b"], mf["b"], cr-1.8, cr+1.8, 1),
            (med["z_b"], mf["b"], ct-1.1, ct+1.1, 1)]
    ai, ar, at, bi, br, bt = [peak_amplitude(*a, degree=degree) for a in args]
    if exact_incident:
        ai = bi = .5
    r, t, rb, tb = (ar/ai)**2, 2*(at/ai)**2, (br/bi)**2, .5*(bt/bi)**2
    return dict(Ai=ai, Ar=ar, At=at, bi=bi, br=br, bt=bt,
                R=r, T=t, R_B=rb, T_B=tb, dR=r-1/9, dT=t-8/9,
                dsum=r+t-1, dR_B=rb-1/9, dT_B=tb-8/9, dsum_B=rb+tb-1)


def spectral_peaks(h=.01, S=.5, lattice_propagation=True, lattice_interface=True,
                   sync=True, quadrature_count=4097, cutoff=14.):
    """Yee 차분식에 평면파를 대입해 얻은 분산식과 경계 계수를 Fourier 적분한다.

    k는 진공 쪽 공간 파수, q=sin(kh/2), k2=2asin(2q)/h이다.
    경계 E node의 eps=2.5이면 r=(cos(kh/2)-2cos(k2h/2))/den, t=1+r.
    실제 시간 적분과 독립적인 비교 계산이며 연속체 관계로도 바꿀 수 있다.
    경계만/전파만 바꾼 경우는 원인 분석을 위한 가상 비교식이다.
    """
    k = np.linspace(-cutoff, cutoff, quadrature_count)
    weights = np.exp(-k*k/4) / (4*np.sqrt(np.pi)) * (k[1]-k[0])
    weights[[0, -1]] *= .5
    q = np.sin(k*h/2)
    assert np.max(np.abs(2*q)) < 1
    kl = 2*np.arcsin(2*q)/h
    w = 2*np.arcsin(S*q)/(S*h) if lattice_propagation else k
    kt = kl if lattice_propagation else 2*k
    c1, c2 = np.sqrt(1-q*q), np.sqrt(1-4*q*q)
    r = (c1-2*c2)/(c1+2*c2) if lattice_interface else np.full_like(k, -1/3)
    t = 1+r
    temporal = np.sqrt(1-S*S*q*q) if lattice_propagation and sync else np.ones_like(k)

    def field(z, tau, magnetic=False, medium=True):
        phase_i = np.exp(1j*(z[:, None]*k-tau*w))
        phase_l = np.exp(1j*(z[:, None]*k+tau*w))
        if not medium:
            value = (phase_i-phase_l)*temporal if magnetic else phase_i+phase_l
        else:
            phase_r = np.exp(1j*((10-z[:, None])*k-tau*w))*r
            phase_t = np.exp(1j*(5*k+(z[:, None]-5)*kt-tau*w))*t
            value = np.where(z[:, None] < 5,
                             (phase_i-phase_l-phase_r)*temporal if magnetic else phase_i+phase_l+phase_r,
                             2*phase_t*temporal if magnetic else phase_t)
        return np.real(value @ weights)

    amplitudes = []
    for magnetic in (False, True):
        for center, tau, is_med, sign in [(3., 3., False, 1), (0., 10., True, 1 if magnetic else -1), (7.5, 10., True, 1)]:
            # 이론적 봉우리가 격자점 사이에 있을 때도 실제 E/B 격자에 맞춘다.
            shift = .5 if magnetic else 0
            nearest_index = int(round((center+20)/h-shift))
            z = -20+h*(nearest_index+np.arange(-8, 9)+shift)
            amplitudes.append(peak_amplitude(z, field(z, tau, magnetic, is_med), z[1], z[-2], sign))
    ai, ar, at, bi, br, bt = amplitudes
    r, t, rb, tb = (ar/ai)**2, 2*(at/ai)**2, (br/bi)**2, .5*(bt/bi)**2
    return dict(Ai=ai, Ar=ar, At=at, bi=bi, br=br, bt=bt,
                R=r, T=t, R_B=rb, T_B=tb, dR=r-1/9, dT=t-8/9,
                dsum=r+t-1, dR_B=rb-1/9, dT_B=tb-8/9, dsum_B=rb+tb-1)


def interface_transfer_from_update(k, h):
    """닫힌 형태의 계수를 복사하지 않고, 경계 갱신식의 2x2 연립식을 푼다.

    진공의 공간 파수 k에 대응하는 반사·투과 진폭비를 반환한다.
    E 경계점에서 r-t=-1이며, 경계의 Ampere 식을 두 번째 행으로 둔다.
    유전율 평균은 원래 계산과 같은 5/2이다.
    """
    q = np.sin(k*h/2)
    assert np.all(np.abs(2*q) < 1)
    theta1, theta2 = np.arcsin(q), np.arcsin(2*q)
    matrix = np.empty((len(k), 2, 2), dtype=complex)
    matrix[:, 0, 0], matrix[:, 0, 1] = 1, -1
    matrix[:, 1, 0] = np.exp(1j*theta1)
    matrix[:, 1, 1] = 2*np.exp(1j*theta2)-5j*q
    rhs = np.stack((-np.ones_like(k), np.exp(-1j*theta1)), axis=-1)
    sol = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    residual = np.max(np.abs(np.einsum('nij,nj->ni', matrix, sol)-rhs))
    assert residual < 2e-14
    assert np.max(np.abs(sol.imag)) < 2e-14
    return sol[:, 0].real, sol[:, 1].real


def remove_interface_amplitude_error(values, z, h, transmitted, cutoff, limits):
    """실제 수치 파형에 경계 계수의 오차만 역보정한다. 시간 적분은 바꾸지 않는다.

    반사파는 k, 투과파는 k2 공간 파수를 사용하므로 투과파에서는
    sin(kh/2)=sin(k2h/2)/2로 진공 쪽 파수를 먼저 계산한다.
    cutoff는 진공 파수의 절댓값이다. 유효 스펙트럼 밖은 원래대로 둔다.
    """
    packet = np.where((z >= limits[0]) & (z < limits[1]), values, 0.)
    spatial_k = 2*np.pi*np.fft.fftfreq(len(packet), d=h)
    vacuum_k = (2*np.arcsin(.5*np.sin(spatial_k*h/2))/h
                if transmitted else spatial_k)
    active = np.abs(vacuum_k) <= cutoff
    rh, th = interface_transfer_from_update(vacuum_k[active], h)
    correction = np.zeros(len(packet), dtype=complex)
    ratio = (2/3)/th if transmitted else (-1/3)/rh
    correction[active] = (ratio-1)*np.fft.fft(packet)[active]
    change = np.fft.ifft(correction)
    assert np.max(np.abs(change.imag)) < 1e-13
    return change.real


def causal_analysis(vacuum=None, medium=None, output_dir=None):
    """경계 진폭비 보정에 따른 오차 감소를 측정하고 검증 결과를 저장한다.

    전체 실행에서는 앞서 계산한 vacuum과 medium을 재사용한다.
    검증만 실행할 때는 필요한 시각의 수치해를 직접 계산한다.
    기존 error_validation/results.json이 있으면 기준값도 대조한다.
    """
    source = Path(__file__).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if vacuum is None:
        vacuum = simulate(snapshot_tau=(3.,))
    if medium is None:
        medium = simulate(dielectric=True, snapshot_tau=(0., 10.))
    baseline = measure_error_coefficients(vacuum, medium)
    reference_path = ROOT / "error_validation" / "results.json"
    if reference_path.is_file():
        stored = json.loads(reference_path.read_text(encoding="utf-8"))
        for key in ('Ai','Ar','At','R','T','R_B','T_B'):
            assert baseline[key] == stored['baseline'][key]

    variants = []
    for cutoff in (10., 12., 14., 16.):
        for refl_limits, tran_limits in [((-5.,5.),(5.,20.)),
                                        ((-4.5,4.5),(5.25,19.))]:
            corrected = dict(medium)
            corrected['snapshots'] = {10.: {}}
            for coord, key in [('z_e','E'), ('z_b','b')]:
                z = medium[coord]
                values = medium['snapshots'][10.][key]
                dr = remove_interface_amplitude_error(values,z,.01,False,cutoff,refl_limits)
                dt = remove_interface_amplitude_error(values,z,.01,True,cutoff,tran_limits)
                corrected['snapshots'][10.][key] = values+dr+dt
            changed = measure_error_coefficients(vacuum, corrected)
            variants.append(dict(cutoff=cutoff, reflected_interval=refl_limits,
                                 transmitted_interval=tran_limits, **changed))
    # 선택한 분석 구간과 주파수 절단값의 영향이 결론을 바꾸지 않는지 검사한다.
    chosen = next(x for x in variants if x['cutoff']==14. and x['reflected_interval']==(-5.,5.))
    stability = {k: max(abs(x[k]-chosen[k]) for x in variants) for k in ('R','T','R_B','T_B')}
    assert max(stability.values()) < 2e-10
    # 시간 갱신과 별도로 유도한 Fourier 합성의 예측과도 비교한다.
    prediction = spectral_peaks(lattice_interface=False)
    mismatch = {k: chosen[k]-prediction[k] for k in ('R','T','R_B','T_B')}
    assert max(abs(v) for v in mismatch.values()) < 2e-10
    reduction = {k: 100*(1-abs(chosen[k])/abs(baseline[k]))
                 for k in ('dR','dT','dR_B','dT_B')}
    assert reduction['dR'] > 99.9 and reduction['dT'] > 99.5
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    data = dict(solver_sha256=digest, baseline=baseline, corrected=chosen,
                error_reduction_percent=reduction, variants=variants,
                stability=stability, independent_prediction=prediction,
                corrected_minus_prediction=mismatch,
                amplitude_errors=dict(Ar=baseline['Ar']+1/6,At=baseline['At']-1/3),
                interpretation='Actual FDTD fields, corrected only by continuum/discrete interface-amplitude ratios. Diagnostic postprocessing; original solver and report results unchanged.')
    folder = (ROOT / "error_validation" if output_dir is None
              else Path(output_dir).resolve())
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'cause_isolation.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:data[k] for k in ('baseline','corrected','error_reduction_percent','stability','corrected_minus_prediction')},indent=2))
    print('Cause-isolation checks passed. Original solver unchanged.')
    print(f"Validation directory: {folder}")
    return data


def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.20,
        "lines.linewidth": 1.8, "savefig.dpi": 180,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def scientific_label(value, decimals=2):
    """그림용 과학적 표기 문자열. 기본값은 계수를 소수 둘째 자리까지 표시한다.

    문자열 변환에만 반올림을 적용한다. 계산 배열에는 영향을 주지 않는다.
    """
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def scientific_tick(value, position):
    """Matplotlib 눈금을 과학적 표기법으로 표시한다. position은 API 인수이다."""
    return "$" + scientific_label(value) + "$"


def field_figure(result, tau, title, dielectric=False, right_only=False):
    """같은 물리적 시각의 수치 전기장 E_x [V/m]와 자기장 B_y [T]를 그린다.

    E는 z_e, B는 반 칸 옮긴 z_b 좌표에 표시한다.
    dielectric/right_only는 result를 생성할 때 사용한 설정과 맞춰 전달한다.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    snap = result["snapshots"][tau]
    axes[0].plot(result["z_e"], snap["E"], color="#176394")
    axes[1].plot(result["z_b"], snap["b"] / C, color="#b65327")
    if dielectric:
        xlim = (-14, 11)
        ylim_e, ylim_b = (-0.24, 1.08), (-1.9, 2.6)
    else:
        xlim = (-6, 6)
        ylim_e = (-0.06, 1.08) if right_only else (-0.04, 0.57)
        ylim_b = (-0.18, 3.65) if right_only else (-1.9, 1.9)
    for ax in axes:
        ax.axhline(0, color="#777777", linewidth=0.5)
        if dielectric:
            ax.axvspan(INTERFACE, xlim[1], color="#c8d8e7", alpha=0.35)
            ax.axvline(INTERFACE, color="#536778", linewidth=1)
    axes[0].set_ylabel(r"$E_x$ [V/m]")
    axes[1].set_ylabel(r"$B_y$ [T]")
    axes[1].set_xlabel(r"$z$ [m]")
    axes[1].set_xlim(*xlim)
    axes[0].set_ylim(*ylim_e)
    axes[1].set_ylim(*(np.array(ylim_b) * 1e-9))
    axes[1].set_xticks([-10, -5, 0, 5, 10] if dielectric else [-6, -3, 0, 3, 6])
    axes[0].set_yticks([-0.2, 0, 0.5, 1.0] if dielectric else
                      ([0, 0.5, 1.0] if right_only else [0, 0.25, 0.5]))
    axes[1].set_yticks(np.array([-1, 0, 1, 2]) * 1e-9 if dielectric else
                      (np.array([0, 1, 2, 3]) * 1e-9 if right_only else
                       np.array([-1.5, 0, 1.5]) * 1e-9))
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_tick))
    if dielectric:
        axes[0].text(0.98, 0.93, r"$\epsilon_r=4.00\times10^0$",
                     ha="right", va="top", transform=axes[0].transAxes, fontsize=8)
    fig.suptitle(title + "\n" + rf"$ct={scientific_label(tau)}$ m, "
                 rf"$t={scientific_label(tau / C, 2)}$ s",
                 fontsize=12, y=0.99)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.12, top=0.86, hspace=0.18)
    return fig


def grid_figure():
    """Yee 격자의 일부를 무차원 좌표로 그려 반 칸·반 스텝 배치를 설명한다.

    축은 실제 위치·시간이 아니라 각각 dz와 dt로 나눈 격자 인덱스이다.
    화살표는 E와 b의 갱신에 필요한 이웃 격자점의 관계를 나타낸다.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    for n in (0, 1, 2):
        ax.scatter(np.arange(4), np.full(4, n), color="#176394", s=45)
    for n in (-0.5, 0.5, 1.5):
        ax.scatter(np.arange(3) + 0.5, np.full(3, n), marker="^", s=45,
                   color="#b65327")
    for x in (1.0, 2.0):
        ax.annotate("", (1.5, 0.5), (x, 0),
                    arrowprops={"arrowstyle": "->", "color": "#74828b"})
    for x in (0.5, 1.5):
        ax.annotate("", (1.0, 1), (x, 0.5),
                    arrowprops={"arrowstyle": "->", "color": "#74828b"})
    ax.set_xticks(np.arange(0, 3.1, 0.5))
    ax.set_xlabel(r"$(z-z_{\min})/\Delta z$ (local index)")
    ax.set_ylabel(r"$t/\Delta t$")
    ax.set_ylim(-0.8, 2.3)
    ax.set_title("Yee grid: space and time are both staggered")
    fig.tight_layout()
    return fig




def save_figure(fig, stem, folder, collection=None):
    """같은 그림을 PDF·PNG로 저장하고, 지정된 경우 모음 PDF에도 추가한다."""
    fig.savefig(folder / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(folder / (stem + ".png"), bbox_inches="tight")
    if collection is not None:
        collection.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_csv(path, rows, columns):
    """지정한 열 순서로 반올림 전의 수치를 CSV에 기록한다."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_simulations(output_dir):
    """세 초기조건을 시간 적분하고 수치장 그래프와 계산값을 저장한다."""
    folder = Path(output_dir).resolve()
    fig_folder = folder / "figures"
    fig_folder.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    # 문제 2의 진공 계산. tau=3은 입사 진폭 측정에도 사용한다.
    vacuum = simulate(snapshot_tau=(0, 2, 3))
    right = simulate(right_only=True, snapshot_tau=(0, 2))
    times_tau = tuple(float(k) for k in range(11))
    medium = simulate(dielectric=True, snapshot_tau=times_tau)
    metrics = measurements(vacuum, medium)
    assert len(medium["snapshots"]) == 11
    assert abs(metrics["R_E"] - 1 / 9) < 2e-4
    assert abs(metrics["T_E"] - 8 / 9) < 2e-4
    assert metrics["invariant_relative_drift"] < 1e-10

    # 문제 2·3의 각 한 쌍과 문제 4의 11쌍을 하나의 PDF에도 저장한다.
    with PdfPages(folder / "all_field_plots.pdf") as collection:
        save_figure(field_figure(vacuum, 2.0, "Problem 2: initially B = 0"),
                    "q2_fields", fig_folder, collection)
        save_figure(field_figure(right, 2.0, "Problem 3: right-going initial field",
                                 right_only=True),
                    "q3_fields", fig_folder, collection)
        for k, tau in enumerate(times_tau):
            save_figure(field_figure(medium, tau,
                                     f"Problem 4: dielectric interface ({k + 1}/11)",
                                     dielectric=True),
                        f"q4_{k:02d}", fig_folder, collection)
    save_figure(grid_figure(), "yee_grid", fig_folder)

    # E와 B 각각의 격자 위치와 계산 배열의 정밀도를 그대로 저장한다.
    np.savez_compressed(
        folder / "fields.npz", z_e=medium["z_e"], z_b=medium["z_b"],
        tau=np.array(times_tau), time_seconds=np.array(times_tau) / C,
        q2_E=vacuum["snapshots"][2.0]["E"],
        q2_B=vacuum["snapshots"][2.0]["b"] / C,
        q3_E=right["snapshots"][2.0]["E"],
        q3_B=right["snapshots"][2.0]["b"] / C,
        q4_E=np.stack([medium["snapshots"][t]["E"] for t in times_tau]),
        q4_B=np.stack([medium["snapshots"][t]["b"] / C for t in times_tau]))
    time_rows = [{"k": k, "ct_m": tau, "t_ns": tau / C * 1e9,
                  "energy_J_per_m2": medium["snapshots"][tau]["energy"]}
                 for k, tau in enumerate(times_tau)]
    save_csv(folder / "times.csv", time_rows,
             ["k", "ct_m", "t_ns", "energy_J_per_m2"])
    summary = {
        "constants": {"c": C, "mu0": MU0, "eps0": EPS0, "sigma": SIGMA},
        "grid": {"zmin": ZMIN, "zmax": ZMAX, "dz": MAIN_DZ,
                 "dt_s": medium["dt"], "Courant": COURANT,
                 "E_nodes": len(medium["z_e"]), "B_nodes": len(medium["z_b"])},
        "q4": metrics,
        "theory": {"r_E": -1 / 3, "t_E": 2 / 3, "R": 1 / 9, "T": 8 / 9,
                   "return_time_ns": 10 / C * 1e9},
    }
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "Yee FDTD: all checks passed.",
        f"Grid: dz={MAIN_DZ:g} m, dt={medium['dt']:.12e} s, S={COURANT:g}",
        f"Problem 2/3 time: 2/c = {2/C*1e9:.9f} ns",
        f"Problem 4 return time: 10/c = {10/C*1e9:.9f} ns",
        "Amplitudes: signed E [V/m], signed B [T]",
    ]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.12g}")
    lines.append("Theory: r=-1/3, t=2/3, R=1/9, T=8/9.")
    lines.append("R and T use the incident right-going packet, Ai~0.5 V/m.")
    lines.append("T_E includes n2/n1=2; T_B includes v2/v1=1/2.")
    lines.append("Q4 snapshots: k=0,...,10; t=k/c seconds.")
    for row in time_rows:
        lines.append(f"k={row['k']:2d}, ct={row['ct_m']:2g} m, t={row['t_ns']:.9f} ns")
    text = "\n".join(lines) + "\n"
    (folder / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Output directory: {folder}")
    return vacuum, medium


def main(argv=None):
    """수치 계산·그래프 생성과 오차 검증을 하나의 명령으로 실행한다."""
    parser = argparse.ArgumentParser(
        description="Yee FDTD 계산·그래프 생성과 반사·투과 오차 검증.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "wavepacket_results",
                        help="수치 계산과 그래프를 저장할 폴더")
    parser.add_argument("--validation-dir", type=Path,
                        default=ROOT / "error_validation",
                        help="오차 검증 결과를 저장할 폴더")
    parser.add_argument("--causal-only", action="store_true",
                        help="그래프 생성 없이 경계 진폭비 오차 검증만 실행")
    args = parser.parse_args(argv)
    if args.causal_only:
        causal_analysis(output_dir=args.validation_dir)
    else:
        vacuum, medium = run_simulations(args.output_dir)
        causal_analysis(vacuum, medium, output_dir=args.validation_dir)


if __name__ == "__main__":
    main()
