import os

import time




from util import *  # util.py 안에 정의된 함수들을 불러옵니다
from dotenv import load_dotenv

load_dotenv()

# ────────────── 설정 ──────────────
Testmode    = True
USER_EMAIL   = os.getenv("GMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_PASS")
QUERY        = "larger:10K"

def main():
    service = get_gmail_service()
    nextPageToken = None
    total_threads = 0

    while True:
        response = service.users().threads().list(
            userId='me',
            q=QUERY,
            maxResults=500,
            pageToken=nextPageToken
        ).execute()
        threads = response.get('threads', [])
        if not threads:
            break

        print(f"{len(threads)} threads found for processing in this page.")
        total_threads += len(threads)

        for thread in threads:
            tid = thread['id']
            thread_data = service.users().threads().get(
                userId='me', id=tid, format='full'
            ).execute()
            print(f"\n[스레드 처리중: {tid}] ({len(thread_data['messages'])} 통)")

            for msg_meta in thread_data['messages']:
                msg_id = msg_meta['id']

                # 메타 정보 가져오기 (크기 및 라벨)
                msg_meta_simple = service.users().messages().get(
                    userId='me', id=msg_id, format='metadata'
                ).execute()
                size   = msg_meta_simple.get("sizeEstimate", 0)
                labels = msg_meta_simple.get("labelIds", [])
                is_in_trash = 'TRASH' in labels

                # 용량 정보 출력
                print(f"    [INFO] 메일 크기: {size / 1024:.1f} KB")
                if not is_in_trash:
                    print("    → ✅ 이 메일은 현재 Gmail 용량을 차지합니다.")
                else:
                    print("    → ⚠️ 이 메일은 휴지통에 있어 곧 삭제 예정입니다.")

                # 원본 메시지 파싱
                msg_raw_obj = service.users().messages().get(
                    userId='me', id=msg_id, format='raw'
                ).execute()
                msg_obj = fetch_rfc822_message_from_raw(msg_raw_obj['raw'])
                subject = decode_mime_words(msg_obj.get("Subject", "(제목 없음)"))
                print(f"  [메일] {subject}")

                # 첨부 파일 목록 수집
                attachments = [
                    p for p in msg_obj.walk()
                    if 'attachment' in p.get("Content-Disposition", "").lower()
                ]

                # 첨부 이미지 요약 (BLIP 캡션 + OCR)
                attach_summaries = summarize_attachments_removing_images(attachments)

                # 인라인 이미지 처리 (본문에서 <img>를 요약 블록으로 치환, BLIP 캡션 + OCR)
                new_html_body, inline_summaries = process_body_with_inline_images_and_remove(msg_obj)

                # 재전송: 이미지 제거된 새 HTML과 요약 텍스트를 함께 전송
                resend_mail_removing_images(
                    msg_obj,
                    attach_summaries,
                    inline_summaries,
                    USER_EMAIL,
                    APP_PASSWORD
                )

                # Testmode인 경우 첫 메일 처리 후 즉시 종료
                if Testmode:
                    print("    [TESTMODE] 한 번만 실행하도록 종료합니다.")
                    return

                # 실제 모드인 경우 메일을 휴지통으로 이동
                try:
                    service.users().messages().trash(
                        userId='me', id=msg_id
                    ).execute()
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
