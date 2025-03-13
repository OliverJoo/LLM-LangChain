import base64
import streamlit as st
from PIL import Image
import re
from openai import OpenAI, BadRequestError
import os
import logging

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI 클라이언트 초기화 ---
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception as e:
    logger.error(f"OpenAI 클라이언트 초기화 오류: {e}")
    st.error("OpenAI 클라이언트 초기화에 실패했습니다. API 키를 확인하세요.")
    exit()


# --- OpenAI를 사용하여 영수증 정보 추출 함수 ---
def extract_receipt_info(image_bytes):
    """
    OpenAI의 비전 API를 사용하여 이미지에서 영수증 정보를 추출합니다. 특히 코스트코 영수증에 맞게 조정되었습니다.

    Args:
        image_bytes (bytes): 이미지 데이터(바이트).

    Returns:
        str: 이미지에서 추출된 텍스트. 오류 발생 시 None을 반환합니다.
    """
    logger.info("OpenAI의 비전 API로 코스트코 영수증 정보 추출 시작...")
    try:
        base64_image = base64_encode(image_bytes)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """이 코스트코 영수증 정보를 다음 형식으로 정확히 추출해주세요:

상호명: (지점명 포함, 예: 코스트코 양재점)
거래일시: (영수증에 있는 날짜와 시간)
구매한 상품 목록:
- 상품명1: 가격1
- 상품명2: 가격2 (할인 적용시 -할인가격)
(모든 상품 나열)
총 금액: (합계 금액)
지불 수단: (결제 방법, 예: 신용카드, 현금 등)
결제 승인 번호: (있을 경우 승인번호)

주의사항: 코스트코 영수증에서는 동일 제품이 연속으로 표시되는 경우가 있습니다. 
첫 번째 항목은 정가이고, 바로 아래에 같은 상품이 나오면서 '-T'로 끝나는 경우 이는 할인 금액입니다. 
이 경우 실제 상품 가격은 '정가 - 할인가'로 계산해야 합니다.
예: 프로쉬주방세제 16,990 다음에 프로쉬주방세제 2,500-T가 나오면 실제 가격은 14,490원입니다.

영수증에서 찾을 수 없는 정보는 "찾을 수 없음"으로 표시해주세요.
"""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        if response and response.choices and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            logger.info("코스트코 영수증 정보 추출 성공.")
            return extracted_text
        else:
            logger.error("코스트코 영수증 정보 추출에 실패하였습니다.")
            st.error("코스트코 영수증 정보 추출에 실패하였습니다.")
            return None
    except BadRequestError as e:
        logger.error(f"코스트코 영수증 정보 추출 중 BadRequestError: {e}")
        st.error(f"요청 오류 발생: {e}")
        return None
    except Exception as e:
        logger.error(f"코스트코 영수증 정보 추출 중 오류: {e}")
        st.error(f"코스트코 영수증 정보 추출 중 예상치 못한 오류 발생: {e}")
        return None


def base64_encode(image_bytes):
    """이미지 바이트를 base64 문자열로 인코딩합니다."""
    return base64.b64encode(image_bytes).decode("utf-8")


# --- 지출 분류 함수 ---
def classify_expense(expense_details, store_name):
    """
    OpenAI의 ChatCompletion API를 사용하여 지출을 여러 카테고리 중 하나로 분류합니다.

    Args:
        expense_details (list): 지출 세부 정보 목록.
        store_name (str): 상점 이름.

    Returns:
        str: 분류된 지출 카테고리.
    """
    logger.info("지출 분류 시작...")
    try:
        # 상점명과 지출 세부 정보 결합
        classification_text = f"Store: {store_name}\nItems: {', '.join(expense_details)}"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "지출 내역이나 상점 이름을 읽고 다음 카테고리 중 하나로 분류하세요: 생활비, 주거비, 교육/보육비, 교통비, 통신비, 문화/여가비, 기타 지출. 식료품이면 '생활비 - 식품'으로, 외식이나 배달이면 '생활비 - 외식'으로, 안경이나 렌즈는 '생활비 - 의료'로 분류하세요. 전체 지출을 하나의 카테고리로만 분류해야 합니다. 자세한 정보나 설명은 출력하지 마세요."},
                {"role": "user", "content": classification_text}
            ],
            temperature=0.5,
        )
        if response and response.choices and response.choices[0].message.content:
            classified_expense = response.choices[0].message.content
            logger.info(f"지출 분류 완료. 분류 결과: {classified_expense}")
            return classified_expense
        else:
            logger.error("지출 분류가 유효한 텍스트를 반환하지 않았습니다.")
            st.error("지출 분류가 유효한 텍스트를 반환하지 않았습니다.")
            return None
    except BadRequestError as e:
        logger.error(f"지출 분류 중 BadRequestError: {e}")
        st.error(f"요청 오류 발생: {e}")
        return None
    except Exception as e:
        logger.error(f"지출 분류 중 오류: {e}")
        st.error(f"지출 분류 중 예상치 못한 오류 발생: {e}")
        return None


# --- OpenAI 응답 파싱 함수 ---
def parse_receipt_response(extracted_text):
    """OpenAI의 구조화된 응답을 파싱합니다."""

    # 정규식 패턴을 사용하여 정보 추출
    store_name_match = re.search(r"상호명:\s*(.*?)(?=\n|$)", extracted_text)
    date_match = re.search(r"거래일시:\s*(.*?)(?=\n|$)", extracted_text)
    total_amount_match = re.search(r"총 금액:\s*(.*?)(?=\n|$)", extracted_text)
    payment_method_match = re.search(r"지불 수단:\s*(.*?)(?=\n|$)", extracted_text)
    card_number_match = re.search(r"결제 승인 번호:\s*(.*?)(?=\n|$)", extracted_text)

    # 값 추출 또는 기본값 설정
    store_name = store_name_match.group(1).strip() if store_name_match else "찾을 수 없음"
    date = date_match.group(1).strip() if date_match else "찾을 수 없음"
    total_amount = total_amount_match.group(1).strip() if total_amount_match else "찾을 수 없음"
    payment_method = payment_method_match.group(1).strip() if payment_method_match else "찾을 수 없음"
    card_number = card_number_match.group(1).strip() if card_number_match else "찾을 수 없음"

    # 항목 목록 추출
    items_section_match = re.search(r"구매한 상품 목록:(.*?)(?=총 금액:|$)", extracted_text, re.DOTALL)

    items = []
    if items_section_match:
        items_text = items_section_match.group(1).strip()
        lines = items_text.split('\n')

        # 할인 적용 처리를 위한 변수들
        current_item = None
        current_price = None
        processed_items = []

        for line in lines:
            # 대시나 콜론이 있는 항목 포맷 처리
            item_match = re.search(r"[-•]\s*(.*?)[:：]\s*(.*?)$", line.strip())
            if item_match:
                item_name = item_match.group(1).strip()
                item_price_str = item_match.group(2).strip()

                # 할인 표시 확인
                discount_match = re.search(r"(\d[\d,]*)\s*\(할인\s*(\d[\d,]*)\)", item_price_str)
                if discount_match:
                    # 할인이 이미 계산된 경우
                    original_price = discount_match.group(1).replace(',', '')
                    discount = discount_match.group(2).replace(',', '')
                    try:
                        final_price = int(original_price) - int(discount)
                        items.append(
                            f"{item_name}: {format(final_price, ',')}원 (정가: {original_price}원, 할인: {discount}원)")
                    except ValueError:
                        items.append(f"{item_name}: {item_price_str}")
                else:
                    # 일반 항목 또는 할인이 별도 항목으로 표시된 경우
                    is_discount = "-T" in item_price_str or "할인" in item_name.lower()
                    price_str = re.sub(r'[^\d,]', '', item_price_str)

                    if is_discount and current_item and current_price:
                        # 이전 항목에 대한 할인인 경우
                        try:
                            discount = int(price_str.replace(',', ''))
                            final_price = current_price - discount
                            processed_items.append(current_item)  # 마크 처리된 항목
                            items.append(
                                f"{current_item}: {format(final_price, ',')}원 (정가: {format(current_price, ',')}원, 할인: {format(discount, ',')}원)")
                            current_item = None
                            current_price = None
                        except ValueError:
                            items.append(f"{item_name}: {item_price_str}")
                    else:
                        # 새 항목인 경우
                        try:
                            price = int(price_str.replace(',', ''))
                            if current_item and current_item not in processed_items:
                                # 이전 항목에 할인이 없었다면 그대로 추가
                                items.append(f"{current_item}: {format(current_price, ',')}원")
                            current_item = item_name
                            current_price = price
                        except ValueError:
                            items.append(f"{item_name}: {item_price_str}")
                            current_item = None
                            current_price = None

        # 마지막 항목 처리
        if current_item and current_item not in processed_items:
            items.append(f"{current_item}: {format(current_price, ',')}원")

    return {
        "store_name": store_name,
        "date": date,
        "items": items,
        "total_amount": total_amount,
        "payment_method": payment_method,
        "card_number": card_number
    }


# --- Streamlit UI ---
st.title("영수증 정보 추출기")
st.subheader("코스트코 영수증 분석기")
uploaded_image = st.file_uploader("영수증 이미지 업로드:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # --- 이미지 회전 처리 ---
    exif = image._getexif()
    if exif:
        orientation = exif.get(0x0112)
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)

    st.image(image, caption='업로드된 영수증 이미지', use_column_width=True)

    if st.button("영수증 정보 추출"):
        with st.spinner("영수증 처리 중..."):
            # 파일 내용을 바이트로 읽기
            image_bytes = uploaded_image.getvalue()
            extracted_text = extract_receipt_info(image_bytes)

            if extracted_text:
                st.success("영수증 정보 추출 완료!")

                # 추출된 텍스트 파싱
                parsed_data = parse_receipt_response(extracted_text)

                # 지출 분류
                if parsed_data["items"]:
                    classification_input = parsed_data["items"]
                elif parsed_data["store_name"] != "찾을 수 없음":
                    classification_input = [parsed_data["store_name"]]
                else:
                    classification_input = []

                expense_category = classify_expense(classification_input, parsed_data["store_name"])

                # 출력 포맷팅
                output = ""
                output += f"상점명: {parsed_data['store_name']}\n"
                output += f"거래일시: {parsed_data['date']}\n"

                if parsed_data["items"]:
                    output += "\n구매 상품 목록:\n"
                    for item in parsed_data["items"]:
                        output += f"- {item}\n"

                output += f"\n총 금액: {parsed_data['total_amount']}\n"
                output += f"결제 방법: {parsed_data['payment_method']}\n"
                output += f"결제 승인 번호: {parsed_data['card_number']}\n"
                output += f"지출 카테고리: {expense_category}\n"

                st.code(output, language='plain')

                # 확장 가능한 섹션에 원시 추출 텍스트 표시
                with st.expander("원시 추출 텍스트 보기"):
                    st.code(extracted_text, language='plain')

            else:
                st.error("영수증 정보를 추출할 수 없습니다. 다른 영수증 이미지를 시도해보세요.")