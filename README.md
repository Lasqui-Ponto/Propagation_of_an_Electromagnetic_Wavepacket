# 전자기파와 광학 연습

Propagation of an Electromagnetic Wavepacket

Maxwell 방정식을 Yee 격자와 Leapfrog 시간 적분으로 풀어 진공 파묶음의 전파 및 유전체 경계에서의 반사·투과를 계산합니다.

- `report/main.pdf`: 현재 보고서
- `report/main.tex`, `report/images/`: 보고서 소스와 그림
- `yee_wavepacket.py`: 문제 2·3·4의 수치 계산·그래프 생성과 보고서 4.6의 경계 진폭비 오차 분석
- `error_validation/`: 오차 분석의 기준 데이터와 계산 결과

## 실행

Python 3에서 NumPy와 Matplotlib을 설치하고 실행합니다.

```sh
python -m pip install -r requirements.txt
python yee_wavepacket.py
```

기본 계산 결과는 `wavepacket_results/`에 저장됩니다. 오차 원인 분석 결과는 `error_validation/cause_isolation.json`에 저장됩니다.

오차 원인 분석만 실행하려면 다음 명령을 사용합니다.

```sh
python yee_wavepacket.py --causal-only
```

`--output-dir`로 계산·그래프 저장 폴더를, `--validation-dir`로 오차 분석 저장 폴더를 지정할 수 있습니다. 전체 실행에서는 수치해를 오차 분석에서도 재사용합니다. 기존 `error_validation/results.json`이 있으면 기준값도 대조하며, 해당 파일이 없어도 독립적인 Fourier 예측과 오차 감소 검증을 수행합니다.

## 보고서 빌드

한국어 TeX 환경과 Nanum 글꼴을 갖춘 TeX Live에서 `report/`로 이동하여 pdfLaTeX를 두 번 실행합니다.

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

이 저장소에는 현재 작업 상태를 보관합니다. 과제 제출 전에는 코드 첫머리의 AI 질문 기록과 PDF의 기록을 일치시켜야 합니다.
