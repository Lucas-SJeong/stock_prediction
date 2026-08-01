graph TD
    
    subgraph Phase 1: 모델 학습 과정 (Jupyter Notebook 또는 스케줄러)
        A[과거 데이터 불러오기\nProcessed_NASDAQ.csv] --> B[데이터 스케일링 및 시퀀스 변환\nseq_len=60]
        B --> C[CNN-LSTM 하이브리드 모델 학습\nEpoch 50]
        C --> D[학습된 모델 및 스케일러 저장\n.keras, .pkl]
    end

    subgraph Phase 2: 실시간 대시보드 예측 과정 (Dash 웹 앱)
        E[Dash 서버 구동 시\n사전 저장된 모델/스케일러 1회 로드]
        D -. 파일 전달 .-> E
        E --> F[60초 주기 타이머 작동]
        F --> G[최근 60일치 데이터 실시간 수집 및 취합\nyfinance 등 활용]
        G --> H[미리 로드한 스케일러로 데이터 전처리]
        H --> I[모델 예측 수행\nInference]
        I --> J[대시보드 UI에 예측 결과 업데이트]
        J -. 60초 후 반복 .-> F
    end
