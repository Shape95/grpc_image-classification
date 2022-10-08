# 개발 환경

1. Visual Studio Code, Anaconda
2. Python 3.8

# 실행방법
[gRPC 통신 proto파일 컴파일]
```bash
python -m grpc.tools.protoc --proto_path=./ --python_out=./ --grpc_python_out=./ classification.proto
```
[라이브러리 다운로드]
```bash
pip install -r requirements.txt
```

3. 동작확인

```bash
# Server-side
python server.py

# Client-side
python client.py
```

# Docker
```bash
@ Dockerhub 

- https://hub.docker.com/r/dlwl963z/server-docker

@ Docker Pull Command

- docker pull dlwl963z/server-docker
```

# 코드
```bash
@ 모델 테스트

- model_quantization.ipynb (resnet 모델 최적화 테스트)

@ Model 파일

- resnet.py
```
# 폴더 구조 
```bash
@ proto

- .proto files

@ script (models)

- .pt, .ptl files 
- model_env_test.ipynb 테스트 중 저장된 모델 리스트

@ deploy
- model-store : .mar files
- config.properties (server 설정 파일)
- 환경 구축에 사용한 명령어 txt 파일 기록

@ image
- Client-Side에서 전송할 이미지 파일 저장

@ script
- ML Model Checkpoint files
- 양자화로 파일 크기나 처리 속도 향상

@ model
- ML Model
```