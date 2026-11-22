# 다자바 DL

거래별 점수를 계산하는 GINe B 추론 패키지다. 각 대상 거래 시각 T 기준 송·수신 양쪽 계좌의 `[T−24시간, T)` 이력을 입력으로 사용한다.

## 개발 아키텍처 / CI 준비

![다자바 DL의 거래 그래프 추론과 CI 계획](docs/images/dl-architecture-ci.png)

그래프는 최근 이력 16건을 사용한다. 방향별 입·출금 건수는 전체 24시간 창에서 계산한다. 거래 점수와 별도 경보 순위 비교 도구는 원본 파일명과 헤더 제외 첫 데이터 행부터 센 번호로 연결한다. 보존된 전수 추론은 로컬 검증이다. GitHub CI 구성은 후속 작업이다.

## 설치 및 실행

Python 3.9의 새 가상환경에서 다음을 실행한다.

~~~sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
python -m pip install . pytest
python -m pip check
python -m pytest -q
dajaba-dl-score --help
~~~

승인된 내부 체크포인트와 해당 정규화 계약을 함께 전달받아 경로를 지정한다.

~~~sh
dajaba-dl-score --checkpoint ./private-models/best.pt --config ./private-models/dl_contract.json --transactions ./local-inputs/transactions.csv --coverage ./local-inputs/coverage.csv --output ./local-results --device cpu
~~~

[입력 계약 및 모델 확인](docs/INFERENCE.md)을 참고한다. 합성 테스트는 자체 체크포인트를 생성한다. 이 저장소에는 실제 가중치·학습 정규화 값·거래 데이터가 포함되지 않는다.
