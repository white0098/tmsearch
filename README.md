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
* 임예림 : Data Argument / 성능 테스트

## ⚙️ 기술 스택 및 파이프라인
___

* DB 처리
  - 라벨링 형식 : JSON or CSV
  - Argumentation Method : Rotation, Shift, Elastic deformation
* 베이스 라인 - 유사도 측정
  - 언어 : `Python 3.9`
  - 이미지 : CNN
  - 자연어 : TF-IDF / 코사인 유사도
* 학습 서버
  - Colab 및 구글 드라이브 연동
* 성능 테스트
  - human evaluation
 
## 📌 주요 기능
---
