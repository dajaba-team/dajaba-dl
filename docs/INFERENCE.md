# DL 추론 계약

- `transactions.csv`에는 `source_file,source_row_number,ts,from_account_key,to_account_key,amount_received,currency_received,payment_format`이 필요하다. 대상 거래와 양쪽 계좌의 전체 이력을 제공한다.
- `coverage.csv`에는 대상 원본 키와 `history_start,history_end,history_complete,sender_complete,receiver_complete`가 필요하다. 원본 키는 파일명과 헤더를 뺀 첫 행부터 센 번호다. 완료 플래그 3개와 실제 시간 경계가 일치해야 한다.
- 대상보다 앞선 분만 이력에 포함한다. 동일 시각 이력의 최신 순서는 원본 행 번호로 결정한다. 검증된 빈 창과 불완전한 창은 구분한다.
- CLI는 새 출력 폴더의 `transaction_scores.csv`와 `result.json`을 쓴다. 경보 순위 결합과 공식 HIGH 판정은 CLI 외부 범위다.

고정 최종 모델 ID는 `HI-Medium-B-GINe-counts-best-13`이다. 내부 제공 파일의 SHA-256은 `fbdb0753ee7208bc4e62b8ecdc39aea6f345f93252f0a3171d340f4f1652c061`이다. 추론기는 계약에 기록된 해시와 실제 체크포인트를 비교한다. [공개 합성 설정](../config/synthetic.example.json)은 테스트에만 쓴다. 모델 구조의 출처는 [IBM Multi-GNN 고정 커밋](https://github.com/IBM/Multi-GNN/tree/252b0252afca109d1d216c411c59ff70753b25fc)이며 Apache 2.0 고지를 유지한다.
