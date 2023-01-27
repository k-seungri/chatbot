#  seq2seq

- 시퀀스 형태의 입력값을 시퀀스 형태의 출력으로 만들 수 있게 하는 모델
- 재귀 순환 신경망(RNN) 모델을 기반
- 인코더(Encoder) 부분과 디코더(Decoder) 부분으로 나눔


<img width="400" alt="seq2seq" src="https://user-images.githubusercontent.com/37443061/214788366-87bec47d-7ca6-4b2f-bd15-419be0c9bfc4.png">


# Project Structure

    .
    ├── data_in                     # 데이터가 존재하는 영역
        ├── ChatBotData.csv         # 전체 데이터
        ├── ChatBotData.csv_short   # 축소된 데이터 (테스트 용도)
        ├── README.md               # 데이터 저자 READMD 파일
    ├── data_out                    # 출력 되는 모든 데이터가 모이는 영역
        ├── vocabularyData.voc      # 사전 파일
        ├── check_point             # check_point 저장 공간
    ├── configs.py                  # 모델 설정에 관한 소스
    ├── data.py                     # data 전처리 및 모델에 주입되는 data set 만드는 소스
    ├── main.py                     # 전체적인 프로그램이 시작되는 소스
    ├── model.py                    # 모델이 들어 있는 소스
    └── predict.py                  # 학습된 모델로 실행 해보는 소스    