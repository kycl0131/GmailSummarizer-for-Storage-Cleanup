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

# ────────────── 환경설정 ──────────────
USER_EMAIL   = "kycl0131@yonsei.ac.kr"
APP_PASSWORD = "ybmu hajp jyhh aquh"
# QUERY        = "larger:1M"
QUERY = "in:anywhere"

SCOPES       = ['https://mail.google.com/']

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

def delete_all_resend_mails():
    service = get_gmail_service()
    query = 'subject:"[재전송 요약]"'
    nextPageToken = None
    deleted_count = 0
    while True:
        response = service.users().messages().list(userId='me', q=query, maxResults=500, pageToken=nextPageToken).execute()
        messages = response.get('messages', [])
        if not messages:
            break
        for msg in messages:
            service.users().messages().trash(userId='me', id=msg['id']).execute()
            deleted_count += 1
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
    print(f"총 {deleted_count}개의 [재전송 요약] 메일을 삭제했습니다.")
if __name__ == '__main__':
    delete_all_resend_mails()
