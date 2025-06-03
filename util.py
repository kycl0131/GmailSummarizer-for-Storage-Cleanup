import os
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
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import easyocr
import re


# ────────────── Gmail API 인증 ──────────────
def get_gmail_service():
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
    return email.message_from_bytes(base64.urlsafe_b64decode(raw_bytes))

# ────────────── BLIP/OCR/유틸 함수 동일 ──────────────

def decode_mime_words(s: str) -> str:
    if not s:
        return ""
    decoded_fragments = decode_header(s)
    return ''.join([
        fragment.decode(encoding or 'utf-8', errors='replace') if isinstance(fragment, bytes) else fragment
        for fragment, encoding in decoded_fragments
    ])

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to("cuda" if torch.cuda.is_available() else "cpu")
reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

def extract_text_from_image(img_bytes: bytes) -> tuple[str, str]:
    pil_color = Image.open(BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=pil_color, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True).strip().lower()
    if any(word in caption for word in ["logo", "symbol", "emblem", "seal"]):
        return "", caption
    img_array = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("이미지 디코딩 실패")
    results = reader.readtext(bgr, detail=1, paragraph=False)
    lines = []
    for box, text, conf in results:
        y_center = (box[0][1] + box[2][1]) / 2
        lines.append((y_center, box[0][0], text))
    lines.sort()
    grouped_lines, current_group, threshold = [], [], 15
    for i, (y, x, text) in enumerate(lines):
        if not current_group:
            current_group.append((x, text, y))
            continue
        _, _, last_y = current_group[-1]
        if abs(y - last_y) <= threshold:
            current_group.append((x, text, y))
        else:
            grouped_lines.append(current_group)
            current_group = [(x, text, y)]
    if current_group:
        grouped_lines.append(current_group)
    grouped_lines_text = []
    for line in grouped_lines:
        line.sort()
        line_text = " ".join([t for x, t, _ in line])
        grouped_lines_text.append(line_text)
    ocr_text = "\n".join(grouped_lines_text)
    return ocr_text, caption

def extract_plain_text(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                charset = part.get_content_charset() or 'utf-8'
                return part.get_payload(decode=True).decode(charset, errors='replace')
        return ""
    else:
        charset = msg.get_content_charset() or 'utf-8'
        return msg.get_payload(decode=True).decode(charset, errors='replace')

def process_body_with_inline_images_and_remove(msg: email.message.Message) -> tuple[str, list]:
    """본문에서 인라인 이미지를 삭제하고, 대신 요약 설명만 남긴다. 요약정보 반환"""
    html_content = None
    html_charset = None
    inline_summaries = []

    # HTML 본문 추출
    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            html_charset = part.get_content_charset() or 'utf-8'
            html_content = part.get_payload(decode=True).decode(html_charset, errors='replace')
            break
    if html_content is None:
        return extract_plain_text(msg), []

    cid_map: dict[str, tuple[str, str]] = {}
    for part in msg.walk():
        content_type = part.get_content_type()
        content_id   = part.get("Content-ID")
        if content_type.startswith("image/") and content_id:
            cid = content_id.strip("<>")
            payload_bytes = part.get_payload(decode=True)
            try:
                ocr_text, caption = extract_text_from_image(payload_bytes)
                cid_map[cid] = (ocr_text, caption)
            except Exception as e:
                cid_map[cid] = (f"OCR 실패: {str(e)}", f"캡션 실패: {str(e)}")

    # <img src="cid:...">를 설명 div로 대체
    def replace_img_tag(match):
        cid = match.group('cid')
        if cid in cid_map:
            ocr_text, caption = cid_map[cid]
            formatted_ocr = ocr_text.replace("\n", "<br>")
            inline_summaries.append(
                f"[인라인 이미지]\n - 캡션: {caption}\n - OCR: {ocr_text}"
            )
            return (
                f"<div style='margin-bottom:1em;'>"
                f"<strong>[인라인 이미지 설명]</strong><br>"
                f"<u>캡션:</u> {caption}<br>"
                f"<u>OCR:</u><br>{formatted_ocr}"
                f"<br><hr></div>"
            )
        else:
            return "[이미지 제거됨]"
    img_pattern = re.compile(r'<img[^>]+src=["\']cid:(?P<cid>[^"\']+)["\'][^>]*>', re.IGNORECASE)
    new_html = img_pattern.sub(replace_img_tag, html_content)
    return new_html, inline_summaries

def summarize_attachments_removing_images(parts: list[email.message.Message]) -> list[str]:
    """첨부파일 중 이미지: BLIP/OCR 요약, 비이미지: 이름만 남김"""
    summaries: list[str] = []
    for part in parts:
        content_disposition = part.get("Content-Disposition", "")
        if content_disposition and 'attachment' in content_disposition.lower():
            filename      = part.get_filename()
            decoded_name  = decode_mime_words(filename or "(이름 없음)")
            content_type  = part.get_content_type()
            payload_bytes = part.get_payload(decode=True)
            if content_type.startswith("image/"):
                try:
                    ocr_text, caption = extract_text_from_image(payload_bytes)
                    summaries.append(
                        f"[첨부 이미지: {decoded_name}]\n"
                        f" - 캡션: {caption}\n"
                        f" - OCR: {ocr_text}\n"
                    )
                except Exception as e:
                    summaries.append(f"[첨부 이미지: {decoded_name}]\n - 설명 실패: {str(e)}\n")
            else:
                # ★★★ 실제 파일 첨부하지 않고, 이름만 요약에 남김!
                summaries.append(f"[첨부 파일: {decoded_name}]")
    return summaries



def clean_duplicate_blocks(body_html):
    # [재전송 요약] <div>~</div> 블록, <b>~</b>, [원본 ...], 요약구역 모두 삭제
    body_html = re.sub(r'(<div[^>]*>\s*\[재전송 요약\][^<]*</div><br>\s*)+', '', body_html, flags=re.I)
    body_html = re.sub(r'(<div><b>\s*\[재전송 요약\][^<]*</b></div><br>\s*)+', '', body_html, flags=re.I)
    body_html = re.sub(r'<b>\s*\[재전송 요약\][^<]*</b>', '', body_html, flags=re.I)
    body_html = re.sub(r'(\[원본 From:.*?\]<br>\s*)+', '', body_html, flags=re.I)
    body_html = re.sub(r'(\[원본 To:.*?\]<br>\s*)+', '', body_html, flags=re.I)
    body_html = re.sub(r'(\[원본 Cc:.*?\]<br>\s*)+', '', body_html, flags=re.I)
    body_html = re.sub(r'(\[원본 Bcc:.*?\]<br>\s*)+', '', body_html, flags=re.I)
    # 이미지/첨부 요약 구역 전체 삭제 (여러 번 있을 수 있음)
    body_html = re.sub(
        r'<hr><div><strong>===== 이미지 요약 =====</strong>.*?</div>', '', body_html, flags=re.I|re.S
    )
    # 중복 <br> 정리
    body_html = re.sub(r'(<br>\s*){2,}', '<br>', body_html, flags=re.I)
    return body_html.strip()

def resend_mail_removing_images(msg: email.message.Message, attach_summaries: list[str], inline_summaries: list[str]) -> None:
    import re
    subject_raw = decode_mime_words(msg.get("Subject", "(제목 없음)"))
    # 제목 중복 완전 제거
    subject = re.sub(r'(\[재전송 요약\][ ]*)+', '', subject_raw).strip()
    subject = "[재전송 요약] " + subject

    from_addr = decode_mime_words(msg.get("From", ""))
    to_addr   = decode_mime_words(msg.get("To", ""))
    cc_addr   = decode_mime_words(msg.get("Cc", ""))
    bcc_addr  = decode_mime_words(msg.get("Bcc", ""))

    top_info = "<div><b>"
    top_info += f"[원본 From: {from_addr}]<br>"
    top_info += f"[원본 To: {to_addr}]<br>"
    if cc_addr:
        top_info += f"[원본 Cc: {cc_addr}]<br>"
    if bcc_addr:
        top_info += f"[원본 Bcc: {bcc_addr}]<br>"
    top_info += "</b></div><br>"

    # 요약구역 준비
    summary_html = ""
    summaries_all = (attach_summaries or []) + (inline_summaries or [])
    if summaries_all:
        summary_html += "<hr><div><strong>===== 이미지 요약 =====</strong><br>"
        for s in summaries_all:
            summary_html += s.replace("\n", "<br>") + "<br>"
        summary_html += "</div>"

    # 본문(이미지/인라인 제거) 후 중복 안내/요약구역/첨부구역 모두 삭제!
    new_html_body, _ = process_body_with_inline_images_and_remove(msg)
    new_html_body = clean_duplicate_blocks(new_html_body)

    # 맨 위에 안내/요약/본문 딱 한 번만
    full_html = f"<div><b>[재전송 요약] {subject}</b></div><br>{top_info}{new_html_body}{summary_html}"

    message_id    = msg.get("Message-ID")
    references    = msg.get("References")
    in_reply_to   = msg.get("In-Reply-To")

    new_msg = MIMEMultipart("alternative")
    new_msg['Subject'] = subject
    new_msg['From']    = USER_EMAIL
    new_msg['To']      = USER_EMAIL

    if message_id:
        new_msg['In-Reply-To'] = message_id
        if references:
            if message_id not in references:
                new_msg['References'] = references + " " + message_id
            else:
                new_msg['References'] = references
        else:
            new_msg['References'] = message_id

    new_msg.attach(MIMEText(full_html, 'html', 'utf-8'))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(USER_EMAIL, APP_PASSWORD)
        smtp.send_message(new_msg)
        print("[INFO] 재전송 완료")
