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
from util import *

from dotenv import load_dotenv
load_dotenv()

# ────────────── 환경설정 ──────────────

USER_EMAIL   = os.getenv("GMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_PASS")

QUERY        = "larger:10K"
# QUERY = "in:anywhere"
# QUERY = "in:anywhere -is:important -is:starred -label:VIP"



# ────────────── 메인: 스레드 전체 처리 (이미지 완전 삭제) ──────────────
def main():
    service = get_gmail_service()
    nextPageToken = None
    total_threads = 0
    while True:
        response = service.users().threads().list(
            userId='me', q=QUERY, maxResults=500,
            pageToken=nextPageToken
        ).execute()
        threads = response.get('threads', [])
        if not threads:
            break
        print(f"{len(threads)} threads found for processing in this page.")
        total_threads += len(threads)
        for thread in threads:
            tid = thread['id']
            thread_data = service.users().threads().get(userId='me', id=tid, format='full').execute()
            print(f"\n[스레드 처리중: {tid}] ({len(thread_data['messages'])} 통)")
            for msg_meta in thread_data['messages']:
                msg_id = msg_meta['id']

                # 👉 메타 정보 가져오기 (크기 및 라벨)
                msg_meta_simple = service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
                size = msg_meta_simple.get("sizeEstimate", 0)
                labels = msg_meta_simple.get("labelIds", [])
                is_in_trash = 'TRASH' in labels

                # → 용량 정보 출력
                print(f"    [INFO] 메일 크기: {size / 1024:.1f} KB")
                if not is_in_trash:
                    print("    → ✅ 이 메일은 현재 Gmail 용량을 차지합니다.")
                else:
                    print("    → ⚠️ 이 메일은 휴지통에 있어 곧 삭제 예정입니다.")

                # 👉 원래 메시지 파싱
                msg_raw_obj = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
                msg_obj = fetch_rfc822_message_from_raw(msg_raw_obj['raw'])
                subject = decode_mime_words(msg_obj.get("Subject", "(제목 없음)"))
                print(f"  [메일] {subject}")

                # 첨부/인라인 처리 루프 (기존 코드 그대로)
                attachments = [p for p in msg_obj.walk() if 'attachment' in p.get("Content-Disposition", "").lower()]
                attach_summaries = summarize_attachments_removing_images(attachments)
                new_html_body, inline_summaries = process_body_with_inline_images_and_remove(msg_obj)
                # resend_mail_removing_images(msg_obj, attach_summaries, inline_summaries)
                resend_mail_removing_images(msg_obj, attach_summaries, inline_summaries, USER_EMAIL, APP_PASSWORD)


                try:
                    service.users().messages().trash(userId='me', id=msg_id).execute()
                    print("    [INFO] 휴지통 이동 완료 — ID", msg_id)
                except Exception as e:
                    print("    [ERROR] 휴지통 이동 실패 —", msg_id, e)

                time.sleep(0.2)

        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
    print(f"[총 {total_threads} 스레드 처리 완료]")



if __name__ == '__main__':
    main()
