import os
# 프로토콜 버퍼 호환성 문제 방지
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pickle
import base64
import time
import email
import smtplib
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import torch
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR

import logging
import re

# ────────────── Gmail API 인증 스코프 ──────────────
SCOPES = ['https://mail.google.com/']


def get_gmail_service():
    """
    Gmail API 서비스 객체를 반환합니다.
    token.pickle이 없거나 만료된 경우 OAuth 플로우를 거쳐 인증하고 token.pickle을 생성합니다.
    """
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)


def fetch_rfc822_message_from_raw(raw_bytes):
    """
    Gmail API로 받아온 raw 바이트(base64url로 인코딩된 메시지)를 디코딩하여
    email.message.Message 객체로 변환해 반환합니다.
    """
    return email.message_from_bytes(base64.urlsafe_b64decode(raw_bytes))


def decode_mime_words(s: str) -> str:
    """
    이메일 헤더(Subject, From, To 등)에 있는 MIME-encoded 단어들을 디코딩하여
    유니코드 문자열로 반환합니다.
    """
    if not s:
        return ""
    decoded_fragments = decode_header(s)
    return ''.join([
        fragment.decode(encoding or 'utf-8', errors='replace') if isinstance(fragment, bytes) else fragment
        for fragment, encoding in decoded_fragments
    ])


# ────────────── BLIP 모델 초기화 ──────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)

model_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ────────────── PaddleOCR 엔진 초기화 ──────────────
logging.getLogger("ppocr").setLevel(logging.INFO)
# ocr = PaddleOCR(lang='korean', use_textline_orientation=True, use_angle_cls=True)
# ocr = PaddleOCR(lang='korean', use_textline_orientation=True)


from paddleocr import PaddleOCR
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# OCR 초기화
# ocr_engine = PaddleOCR(use_angle_cls=True, lang='korean', show_log=False)

ocr = PaddleOCR(lang="korean", use_textline_orientation=True, use_angle_cls=True)



def extract_text_from_image(img_bytes: bytes) -> tuple[str, str]:
    """
    입력된 이미지 바이트에서
      1) BLIP을 이용한 캡션(caption)을 생성하고,
      2) PaddleOCR을 이용해 글자를 인식한 뒤
         - 최소한의 전처리(그레이스케일 변환만) 후
         - 검출된 텍스트 블록들을 “줄 단위”로 재구성하여 반환합니다.

    반환: (ocr_text, caption)
      - caption: BLIP이 생성한 이미지 캡션
      - ocr_text: 줄바꿈과 띄어쓰기를 최대한 보존하여 재구성한 OCR 결과
    """
    # ────────────── 1) BLIP 캡션 생성 ──────────────
    try:
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        out = model_caption.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).strip().lower()
        print(f"[DEBUG][BLIP] 최종 캡션: {caption}")
    except Exception as e:
        caption = f"캡션 실패: {e}"
        print(f"[ERROR][BLIP] 캡션 생성 중 예외 발생: {e}")

    # 로고·심볼류 캡션이면 OCR 생략
    if any(keyword in caption for keyword in ["logo", "symbol", "seal"]):
        return "", caption

    # ────────────── 2) OCR 최소 전처리 및 실행 ──────────────
    try:
        # (2-1) PIL → OpenCV BGR 변환
        rgb_img = np.array(pil_img)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # (2-2) 그레이스케일 변환만
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        print(f"[DEBUG][OCR] 그레이스케일 변환 완료, 이미지 크기: {gray.shape}")

        # (2-3) PaddleOCR 호출 (전처리 생략)
        results = ocr.ocr(gray, cls=True)
        print(f"[DEBUG][OCR] Raw Results:\n{results}")

        # (2-4) 검출된 블록들을 (top_y, left_x, text, confidence) 형태로 수집
        candidates = []
        for item in results:
            try:
                bbox, text_info = item
                if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                    text, score = text_info[0], float(text_info[1])
                    # confidence가 0.50 미만이면 제외
                    if score < 0.50:
                        continue
                    stripped = text.strip()
                    # 공백만 있거나 숫자만 있으면 제외
                    if not stripped or stripped.replace(" ", "").isdigit():
                        continue
                    # bbox 네 점의 좌표에서 top_y와 left_x 계산
                    ys = [pt[1] for pt in bbox]
                    xs = [pt[0] for pt in bbox]
                    top_y = min(ys)
                    left_x = min(xs)
                    candidates.append((top_y, left_x, stripped))
                else:
                    continue
            except Exception as e:
                print(f"[WARNING][OCR] 아이템 파싱 오류: {e} — item={item}")
                continue

        # (2-5) y 좌표 기준으로 오름차순 정렬
        candidates.sort(key=lambda x: x[0])

        # (2-6) 같은 “라인”로 묶기
        lines: list[list[tuple[int, int, str]]] = []
        if candidates:
            current_line = [candidates[0]]
            for top_y, left_x, txt in candidates[1:]:
                prev_top_y = current_line[-1][0]
                # “라인 간격” 임계치 (픽셀 단위)
                line_threshold = 20  # 최소 전처리 시 간격이 더 느슨해도 무방
                if abs(top_y - prev_top_y) <= line_threshold:
                    current_line.append((top_y, left_x, txt))
                else:
                    lines.append(current_line)
                    current_line = [(top_y, left_x, txt)]
            lines.append(current_line)
        else:
            lines = []

        # (2-7) 각 줄 안에서 x 순서대로 정렬하고, 중간에 공백 하나씩 삽입
        reconstructed_lines: list[str] = []
        for line_blocks in lines:
            line_blocks.sort(key=lambda x: x[1])
            words = [blk[2] for blk in line_blocks]
            reconstructed_lines.append(" ".join(words))

        ocr_text = "\n".join(reconstructed_lines).strip()
        if not ocr_text:
            ocr_text = "[텍스트 없음]"
        print(f"[DEBUG][OCR] 최종 추출된 텍스트:\n{ocr_text}")

    except Exception as e:
        print(f"[ERROR][OCR] OCR 수행 실패: {e}")
        ocr_text = "[텍스트 없음]"

    return ocr_text, caption







def extract_plain_text(msg: email.message.Message) -> str:
    """
    전달된 email.message.Message 객체에서 텍스트/플레인 본문을 추출하여 반환합니다.
    """
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                charset = part.get_content_charset() or 'utf-8'
                return part.get_payload(decode=True).decode(charset, errors='replace')
        return ""
    else:
        charset = msg.get_content_charset() or 'utf-8'
        return msg.get_payload(decode=True).decode(charset, errors='replace')


def process_body_with_inline_images_and_remove(msg: email.message.Message) -> tuple[str, list[str]]:
    """
    이메일 메시지(msg)의 본문에서 인라인 이미지를 찾아 제거하고,
    각 인라인 이미지에 대해 BLIP 캡션 + OCR을 수행한 요약 정보를 반환합니다.

    반환:
      new_html (str)             : 인라인 이미지가 요약 박스로 치환된 HTML 문자열
      inline_summaries (list[str]): 각 인라인 이미지에 대한 요약 텍스트 리스트
    """
    html_content = None
    html_charset = None
    inline_summaries: list[str] = []

    # (1) HTML 본문 추출
    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            html_charset = part.get_content_charset() or 'utf-8'
            html_content = part.get_payload(decode=True).decode(html_charset, errors='replace')
            break

    if html_content is None:
        # HTML 본문이 없는 경우, 플레인 텍스트를 반환
        return extract_plain_text(msg), []

    # (2) CID → (ocr_text, caption) 매핑
    cid_map: dict[str, tuple[str, str]] = {}
    for part in msg.walk():
        content_type = part.get_content_type()
        content_id = part.get("Content-ID")
        if content_type.startswith("image/") and content_id:
            cid = content_id.strip("<>")
            payload_bytes = part.get_payload(decode=True)
            try:
                ocr_text, caption = extract_text_from_image(payload_bytes)
            except Exception as e:
                ocr_text, caption = "", f"캡션 실패: {e}"
            cid_map[cid] = (ocr_text, caption)

    # (3) <img src="cid:..."> 태그를 요약 박스로 치환 (스타일 적용)
    def replace_img_tag(match):
        cid = match.group("cid")
        if cid in cid_map:
            ocr_text, caption = cid_map[cid]
            formatted_ocr = ocr_text.replace("\n", "<br>")
            inline_html = (
                "<div style='margin:10px 0; padding:10px; "
                "border:1px solid #cccccc; background-color:#f9f9f9; border-radius:4px;'>"
                "<strong>📷 인라인 이미지 요약</strong><br>"
                f"<em>캡션:</em> {caption}<br>"
                f"<em>🔍 OCR:</em><br>{formatted_ocr or '[텍스트 없음]'}"

                "</div>"
            )
            inline_summaries.append(
                f"[인라인 이미지]\n - 캡션: {caption}\n - OCR: {ocr_text}"
            )
            return inline_html
        else:
            return "[이미지 제거됨]"

    img_pattern = re.compile(
        r'<img[^>]+src=["\']cid:(?P<cid>[^"\']+)["\'][^>]*>', re.IGNORECASE
    )
    new_html = img_pattern.sub(replace_img_tag, html_content)
    return new_html, inline_summaries


def summarize_attachments_removing_images(parts: list[email.message.Message]) -> list[str]:
    """
    이메일 파트 리스트(parts)에서 attachment 파트를 순회하며,
    - Content-ID가 있는(인라인) 이미지는 건너뛰고,
    - 나머지 이미지 attachment에 대해서만 BLIP 캡션 + OCR을 수행하여 요약 박스를 생성,
      그 HTML 블록을 리스트에 담아 반환합니다.
    - 비이미지 attachment는 파일명만 텍스트로 반환.

    반환:
      summaries (list[str]) : 각 첨부파일 요약 문자열 (이미지: HTML 박스, 그 외: 파일명)
    """
    summaries: list[str] = []
    for part in parts:
        content_disposition = part.get("Content-Disposition", "")
        content_id = part.get("Content-ID", None)

        # 인라인 이미지(Content-ID 있음)는 이미 처리했으므로 스킵
        if content_id:
            continue

        if not (content_disposition and 'attachment' in content_disposition.lower()):
            continue

        filename = part.get_filename()
        decoded_name = decode_mime_words(filename or "(이름 없음)")
        content_type = part.get_content_type()
        payload_bytes = part.get_payload(decode=True)

        if content_type.startswith("image/"):
            try:
                ocr_text, caption = extract_text_from_image(payload_bytes)
                html_block = (
                    "<div style='margin:10px 0; padding:10px; "
                    "border:1px solid #cccccc; background-color:#f0f0f0; border-radius:4px;'>"
                    f"<strong>📎 첨부 이미지: {decoded_name}</strong><br>"
                    f"<em>캡션:</em> {caption}<br>"
                    f"<em>🔍 OCR:</em><br>{ocr_text.replace(chr(10), '<br>')}"
                    "</div>"
                )
                summaries.append(html_block)
            except Exception as e:
                summaries.append(
                    f"[첨부 이미지: {decoded_name}]\n - 설명 실패: {e}\n"
                )
        else:
            # 비이미지 첨부는 파일명만 텍스트로 남김
            summaries.append(f"[첨부 파일: {decoded_name}]")
    return summaries


def clean_duplicate_blocks(body_html: str) -> str:
    """
    [재전송 요약] 등 중복 블록, 원본 헤더 등 불필요한 마크업을 제거합니다.
    """
    html = body_html
    html = re.sub(r'(<div[^>]*>\s*\[재전송 요약\][^<]*</div><br>\s*)+', '', html, flags=re.I)
    html = re.sub(r'(<div><b>\s*\[재전송 요약\][^<]*</b></div><br>\s*)+', '', html, flags=re.I)
    html = re.sub(r'<b>\s*\[재전송 요약\][^<]*</b>', '', html, flags=re.I)
    html = re.sub(r'(\[원본 From:.*?\]<br>\s*)+', '', html, flags=re.I)
    html = re.sub(r'(\[원본 To:.*?\]<br>\s*)+', '', html, flags=re.I)
    html = re.sub(r'(\[원본 Cc:.*?\]<br>\s*)+', '', html, flags=re.I)
    html = re.sub(r'(\[원본 Bcc:.*?\]<br>\s*)+', '', html, flags=re.I)
    html = re.sub(
        r'<hr><div><strong>===== 이미지 요약 =====</strong>.*?</div>', '', html, flags=re.I | re.S
    )
    html = re.sub(r'(<br>\s*){2,}', '<br>', html, flags=re.I)
    return html.strip()


def resend_mail_removing_images(msg: email.message.Message,
                                attach_summaries: list[str],
                                inline_summaries: list[str],
                                user_email: str,
                                app_password: str) -> None:
    """
    이미지를 제거한 상태로 메일을 재전송합니다.
    - msg: 원본 email.message.Message 객체
    - attach_summaries: summarize_attachments_removing_images()의 결과 HTML 블록 리스트
    - inline_summaries: process_body_with_inline_images_and_remove()의 결과 텍스트 리스트
    - user_email: 보내는 사람(자기 자신의 이메일)
    - app_password: 애플리케이션 비밀번호

    최종 전송 이메일 구성:
      1) [재전송 요약] + 원본 From/To/Cc/Bcc 정보
      2) 인라인 이미지가 요약 박스로 치환된 HTML 본문(new_html_body)
      3) 요약 블록: 각 첨부 이미지 HTML 박스 + 인라인 요약 텍스트(디버그용)
    """
    # (1) 제목 처리
    subject_raw = decode_mime_words(msg.get("Subject", "(제목 없음)"))
    subject = re.sub(r'(\[재전송 요약\][ ]*)+', '', subject_raw).strip()
    subject = "[재전송 요약] " + subject

    # (2) 원본 정보
    from_addr = decode_mime_words(msg.get("From", ""))
    to_addr = decode_mime_words(msg.get("To", ""))
    cc_addr = decode_mime_words(msg.get("Cc", ""))
    bcc_addr = decode_mime_words(msg.get("Bcc", ""))

    top_info = "<div><b>"
    top_info += f"[원본 From: {from_addr}]<br>"
    top_info += f"[원본 To: {to_addr}]<br>"
    if cc_addr:
        top_info += f"[원본 Cc: {cc_addr}]<br>"
    if bcc_addr:
        top_info += f"[원본 Bcc: {bcc_addr}]<br>"
    top_info += "</b></div><br>"

    # (3) 본문 처리 및 중복 제거
    new_html_body, _ = process_body_with_inline_images_and_remove(msg)
    new_html_body = clean_duplicate_blocks(new_html_body)

    # (4) 첨부 요약 블록(이미지) + 인라인 요약 텍스트 모아서 HTML 생성
    summary_html = ""
    if attach_summaries or inline_summaries:
        summary_html += "<hr><div><strong>===== 이미지 첨부 요약 =====</strong><br>"

        # attach_summaries: 이미 HTML 블록 형태이므로 그대로 삽입
        for block in attach_summaries:
            summary_html += block

        # inline_summaries: 디버그용 텍스트 요약이므로 <pre>로 묶어서 표시
        if inline_summaries:
            summary_html += (
                "<div style='margin:10px 0; padding:10px; "
                "border:1px dashed #999999; background-color:#ffffff; border-radius:4px;'>"
                "<strong>🔍 인라인 이미지 요약(텍스트)</strong><br>"
                "<pre style='white-space:pre-wrap; font-family:monospace;'>"
            )
            for txt in inline_summaries:
                summary_html += txt + "\n"
            summary_html += "</pre></div>"

        summary_html += "</div>"

    # (5) 전체 HTML
    full_html = (
        f"<div><b>[재전송 요약] {subject}</b></div><br>"
        f"{top_info}"
        f"{new_html_body}"
        f"{summary_html}"
    )

    # (6) 메시지 헤더 설정
    new_msg = MIMEMultipart("alternative")
    new_msg['Subject'] = subject
    new_msg['From'] = user_email
    new_msg['To'] = user_email

    message_id = msg.get("Message-ID")
    references = msg.get("References")
    if message_id:
        new_msg['In-Reply-To'] = message_id
        if references and message_id not in references:
            new_msg['References'] = references + " " + message_id
        else:
            new_msg['References'] = references or message_id

    new_msg.attach(MIMEText(full_html, 'html', 'utf-8'))

    # (7) SMTP로 전송
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user_email, app_password)
        smtp.send_message(new_msg)
        print("[INFO] 재전송 완료")
