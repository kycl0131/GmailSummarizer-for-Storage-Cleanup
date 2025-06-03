# GmailSummarizer-for-Storage-Cleanup


Gmail API를 활용하여 이메일에 포함된 첨부파일과 인라인 이미지를 자동으로 분석하고 요약.
메일 용량을 줄이기 위해 원본 첨부파일과 이미지 파일은 제거하고, 해당 파일명과 요약 정보를 본문에 포함한 형태로 메일을 재전송.

## 기능

- Gmail API 기반 전체 메일 스레드 탐색
- 이미지 파일의 OCR 및 캡셔닝 (EasyOCR + BLIP 사용) 
- 본문 내 인라인 이미지 분석 및 제거, 설명으로 대체
- 첨부파일 중 이미지에 대한 요약 정보 추출
- 전체 스레드를 유지하며, 원본 내용을 요약 포함 재전송
- 원본 메일은 휴지통으로 이동 (용량 정리 목적)


## 사용 기술

- Gmail API (OAuth2 인증)
- `transformers` – Salesforce BLIP 모델 (이미지 캡셔닝)
- `easyocr` – 이미지 내 텍스트 인식
- `cv2`, `numpy`, `Pillow` – 이미지 처리
- `imaplib`, `smtplib` – 이메일 전송/수신 처리 (혼합)
- Python 3.10+


## 의존 패키지 설치
pip install -r requirements.txt