# Similar Trademark Search
---

## 💻 프로젝트 소개
---
발음, 철자, 이미지를 기반으로한 유사상표 검색 서비스 입니다. 

Spelling, pronunce, and image-based trademark search algorithms by A.I

### ⏰ 개발기간
---

* 2023. 08 ~

### 👨‍👨‍👧 멤버 구성

* 백건희 : Data 수집, 전처리 및 라벨링 / 모델 개발 및 학습
* 정영준 : Data 수집, 전처리 및 라벨링 / 모델 개발 및 학습
* 임예림 : 성능 테스트

## ⚙️ 기술 스택 및 파이프라인
___

* DB 처리
  - 라벨링 형식 : JSON or CSV
  - 벡터 DB: npy
* 베이스 라인 - 유사도 측정
  - 언어 : `Python 3.9`
  - 이미지 : Feature Extactor: VGG, ViT
  - 자연어 : TF-IDF / 코사인 유사도
  - 라이브러리: [faiss](https://github.com/facebookresearch/faiss)
* 학습 서버
  - Colab 및 구글 드라이브 연동
* 성능 테스트
  - human evaluation
  - 비엔나코드 활용
## 📌 주요 기능
---

## 디자인
<img width="50%" src="https://github.com/white0098/tmsearch/assets/12624550/51d80936-25a2-4c19-ad1e-4f8062a1d60e"/>
<img width="50%" src="https://github.com/white0098/tmsearch/assets/12624550/a3f3875f-42f2-4676-b889-ec9bbb20885b"/>
