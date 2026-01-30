# SAM 3D Objects - RunPod Deployment Guide

이 가이드는 SAM 3D Objects 모델을 RunPod에 배포하는 방법을 설명합니다.

## 목차

1. [요구사항](#요구사항)
2. [배포 옵션](#배포-옵션)
3. [Docker 이미지 빌드](#docker-이미지-빌드)
4. [RunPod Serverless 배포](#runpod-serverless-배포)
5. [RunPod GPU Pod 배포](#runpod-gpu-pod-배포)
6. [API 사용법](#api-사용법)
7. [문제 해결](#문제-해결)

---

## 요구사항

### GPU 요구사항
- **최소 32GB VRAM** (A100 40GB, A100 80GB, A6000 권장)
- CUDA 12.1 호환 GPU

### 소프트웨어
- Docker with NVIDIA Container Toolkit
- HuggingFace 계정 및 액세스 토큰 (체크포인트 다운로드용)

### HuggingFace 체크포인트 접근 권한
1. [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) 모델 페이지 방문
2. "Access repository" 버튼 클릭하여 접근 권한 요청
3. 승인 후 HuggingFace 토큰 생성: https://huggingface.co/settings/tokens

---

## 배포 옵션

| 옵션 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Serverless** | 요청당 과금, 자동 스케일링 | 비용 효율적, 관리 불필요 | Cold start 지연 |
| **GPU Pod** | 상시 가동 인스턴스 | 즉시 응답, SSH 접속 가능 | 지속 비용 발생 |

---

## Docker 이미지 빌드

### 로컬에서 빌드

```bash
# 프로젝트 루트에서 실행
cd /path/to/sam-3d-objects

# Docker 이미지 빌드 (약 30-60분 소요)
docker build -t sam3d-objects:latest -f runpod/Dockerfile .

# DockerHub 또는 레지스트리에 푸시
docker tag sam3d-objects:latest your-registry/sam3d-objects:latest
docker push your-registry/sam3d-objects:latest
```

### 빌드 시 주의사항

- PyTorch3D, Flash Attention 컴파일에 시간이 소요됩니다
- CUDA 12.1 환경에서 빌드해야 합니다
- 빌드 머신에 충분한 RAM (16GB+) 필요

---

## RunPod Serverless 배포

### 1. 템플릿 생성

1. [RunPod Console](https://www.runpod.io/console/serverless) 접속
2. "Custom Template" 선택
3. 다음 설정 입력:

```
Container Image: your-registry/sam3d-objects:latest
Container Start Command: /app/start.sh
Container Disk: 50 GB (체크포인트 포함 시)
```

### 2. 환경 변수 설정

```
HF_TOKEN=hf_xxxxxxxxxxxxx  # HuggingFace 토큰
RUN_MODE=serverless
AUTO_LOAD_MODEL=true
```

### 3. GPU 설정

```
GPU Type: A100 40GB 또는 A100 80GB
Max Workers: 1-5 (필요에 따라)
Idle Timeout: 30 (초)
```

### 4. 배포

"Deploy" 클릭 후 엔드포인트 URL 확인

---

## RunPod GPU Pod 배포

### 1. Pod 생성

1. [RunPod Console](https://www.runpod.io/console/pods) 접속
2. "Deploy" 클릭
3. GPU 선택: A100 40GB/80GB 또는 A6000
4. 템플릿 선택: "RunPod Pytorch 2.1" 또는 커스텀 이미지

### 2. SSH 접속 후 설정

```bash
# 프로젝트 클론
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects

# 환경 설정
export HF_TOKEN="hf_xxxxxxxxxxxxx"
export CUDA_HOME=/usr/local/cuda

# Docker 없이 직접 실행하는 경우
pip install -r requirements.txt
pip install -e '.[dev]'
pip install -e '.[p3d]'
pip install -e '.[inference]'

# 체크포인트 다운로드
huggingface-cli login --token $HF_TOKEN
huggingface-cli download --repo-type model --local-dir checkpoints/hf-download facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf

# API 서버 시작
python runpod/api_server.py

# 또는 Gradio 웹 인터페이스
python runpod/gradio_app.py
```

### 3. Volume 마운트 (권장)

체크포인트를 Network Volume에 저장하면 Pod 재시작 시 다시 다운로드할 필요 없음:

```bash
# Volume 마운트 경로: /runpod-volume
mkdir -p /runpod-volume/checkpoints
mv checkpoints/hf /runpod-volume/checkpoints/
```

---

## API 사용법

### Serverless API 호출

```python
import requests
import base64

# 이미지를 base64로 인코딩
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# 마스크 (선택사항)
with open("mask.png", "rb") as f:
    mask_base64 = base64.b64encode(f.read()).decode()

# API 호출
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={
        "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "image": image_base64,
            "mask": mask_base64,  # 선택사항
            "seed": 42,
            "output_format": "ply"  # "ply", "glb", or "both"
        }
    }
)

result = response.json()

# PLY 파일 저장
if "output" in result and "ply" in result["output"]:
    ply_data = base64.b64decode(result["output"]["ply"])
    with open("output.ply", "wb") as f:
        f.write(ply_data)
```

### 직접 API 서버 호출 (GPU Pod)

```python
import requests
import base64

# 이미지 인코딩
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# API 호출
response = requests.post(
    "http://YOUR_POD_IP:8000/predict",
    json={
        "image": image_base64,
        "seed": 42,
        "output_format": "ply"
    }
)

result = response.json()
print(result["status"])
```

### cURL 예제

```bash
# 이미지를 base64로 변환
IMAGE_BASE64=$(base64 -i image.png)

# Serverless 호출
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"image\": \"$IMAGE_BASE64\",
      \"seed\": 42
    }
  }"
```

---

## 응답 형식

```json
{
  "status": "success",
  "ply": "<base64 encoded PLY file>",
  "glb": "<base64 encoded GLB file (optional)>",
  "rotation": [[w, x, y, z]],
  "translation": [[x, y, z]],
  "scale": [[sx, sy, sz]]
}
```

---

## 문제 해결

### 1. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결방법:**
- 더 큰 GPU 사용 (A100 80GB 권장)
- 입력 이미지 크기 줄이기
- 배치 처리 대신 단일 이미지 처리

### 2. 체크포인트 다운로드 실패

```
Error: Model checkpoint not found
```

**해결방법:**
- HF_TOKEN 환경 변수 확인
- HuggingFace 모델 접근 권한 확인
- 수동 다운로드 후 Volume 마운트

### 3. PyTorch3D 컴파일 오류

**해결방법:**
- CUDA 12.1 환경 확인
- GCC 버전 호환성 확인 (GCC 12 권장)
- 미리 빌드된 Docker 이미지 사용

### 4. Cold Start 지연

Serverless에서 첫 요청 시 모델 로딩으로 인한 지연 발생

**해결방법:**
- `AUTO_LOAD_MODEL=true` 설정으로 시작 시 모델 로드
- Idle Timeout 증가
- GPU Pod 사용으로 상시 가동

---

## 비용 예상

### RunPod Serverless
- A100 40GB: ~$0.0013/sec ($4.68/hour)
- A100 80GB: ~$0.0019/sec ($6.84/hour)
- 추론 시간: 약 30-60초/요청

### RunPod GPU Pod
- A100 40GB: ~$1.69/hour (Community Cloud)
- A100 80GB: ~$2.29/hour (Community Cloud)
- A6000 48GB: ~$0.79/hour (Community Cloud)

---

## 파일 구조

```
runpod/
├── Dockerfile          # Docker 이미지 빌드 설정
├── handler.py          # RunPod Serverless 핸들러
├── api_server.py       # Flask REST API 서버
├── gradio_app.py       # Gradio 웹 인터페이스
├── start.sh            # 컨테이너 시작 스크립트
├── docker-compose.yml  # 로컬 테스트용 Compose 설정
└── README.md           # 이 문서
```

---

## 라이선스

이 배포 스크립트는 SAM 3D Objects 프로젝트의 일부로, 해당 프로젝트의 라이선스를 따릅니다.
