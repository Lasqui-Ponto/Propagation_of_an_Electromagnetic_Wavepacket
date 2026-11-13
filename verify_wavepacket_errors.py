"""보고서의 반사·투과 계수 오차를 분석하는 별도 검증 코드.

실행: python verify_wavepacket_errors.py
수치 파형의 경계 진폭비만 보정하여 오차 감소량을 확인한다.
결과는 error_validation/cause_isolation.json에 저장한다.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("original_yee", ROOT / "yee_wavepacket.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = ROOT / "error_validation"
OUT.mkdir(exist_ok=True)




def peak(z, y, lo, hi, sign=1, degree=2):
    if degree == 2:
        return base.peak(z, y, lo, hi, sign)["amplitude"]
    mask = np.flatnonzero((z >= lo) & (z <= hi))
    j = mask[np.argmax(sign * y[mask])]
    radius = degree // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    coeff = np.polynomial.polynomial.polyfit(x, sign * y[j-radius:j+radius+1], degree)
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coeff))
    choices = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and abs(r.real) <= 1]
    assert choices
    return sign * max(np.polynomial.polynomial.polyval(q, coeff) for q in choices)


def measure(vac, med, time=10., degree=2, exact_incident=False):
    vi, mf = vac["snapshots"][3.], med["snapshots"][time]
    cr, ct = 10-time, 5+(time-5)/2
    args = [(vac["z_e"], vi["E"], 2, 4, 1),
            (med["z_e"], mf["E"], cr-1.8, cr+1.8, -1),
            (med["z_e"], mf["E"], ct-1.1, ct+1.1, 1),
            (vac["z_b"], vi["b"], 2, 4, 1),
            (med["z_b"], mf["b"], cr-1.8, cr+1.8, 1),
            (med["z_b"], mf["b"], ct-1.1, ct+1.1, 1)]
    ai, ar, at, bi, br, bt = [peak(*a, degree=degree) for a in args]
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
            amplitudes.append(peak(z, field(z, tau, magnetic, is_med), z[1], z[-2], sign))
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


def causal_analysis():
    """원래 수치해에 경계 진폭비 보정만 적용하고, 오차 감소량을 직접 측정한다."""
    digest = hashlib.sha256((ROOT/'yee_wavepacket.py').read_bytes()).hexdigest()
    vacuum = base.simulate(snapshot_tau=(3.,))
    medium = base.simulate(dielectric=True, snapshot_tau=(0., 10.))
    baseline = measure(vacuum, medium)
    stored = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
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
            changed = measure(vacuum, corrected)
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
    assert hashlib.sha256((ROOT/'yee_wavepacket.py').read_bytes()).hexdigest()==digest
    data = dict(solver_sha256=digest, baseline=baseline, corrected=chosen,
                error_reduction_percent=reduction, variants=variants,
                stability=stability, independent_prediction=prediction,
                corrected_minus_prediction=mismatch,
                amplitude_errors=dict(Ar=baseline['Ar']+1/6,At=baseline['At']-1/3),
                interpretation='Actual FDTD fields, corrected only by continuum/discrete interface-amplitude ratios. Diagnostic postprocessing; original solver and report results unchanged.')
    (OUT/'cause_isolation.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps({k:data[k] for k in ('baseline','corrected','error_reduction_percent','stability','corrected_minus_prediction')},indent=2))
    print('Cause-isolation checks passed. Original solver unchanged.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--causal-only', action='store_true',
                        help='호환용 옵션: 경계 진폭비의 원인 제거 실험을 실행한다.')
    parser.parse_args()
    causal_analysis()
