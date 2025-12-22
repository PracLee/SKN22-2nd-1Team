# app/pages/00_home.py
import streamlit as st
import requests
from io import BytesIO

from ui.header import render_header

# =========================================================
# [STEP 0] Streamlit 기본 설정
# - 반드시 render_header() 호출 전에 위치해야 함
# =========================================================
st.set_page_config(
    page_title="김선생님... 코딩이 하고싶어요...",
    layout="wide"
)

# =========================================================
# [STEP 1] 공통 헤더 렌더링
# =========================================================
render_header()

# =========================================================
# [STEP 2] Google Drive 이미지 로더
# =========================================================
@st.cache_data
def load_image_from_drive(url: str):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return BytesIO(res.content)
    except Exception:
        return None
    return None

# =========================================================
# [STEP 3] 대문 이미지 URL (Google Drive direct view)
# - 공유 설정: 링크가 있는 모든 사용자 "보기 가능"
# - 형식: https://drive.google.com/uc?id=FILE_ID
# =========================================================
HERO_IMAGE_URL = "https://cdn.discordapp.com/attachments/1450006718611329178/1452473129501720686/image.png?ex=6949f083&is=69489f03&hm=56d43d1212e3d46c431615945661e0f3efcff116f46dd1f02b3bb7ce011cb427&"

img_bytes = load_image_from_drive(HERO_IMAGE_URL)

# =========================================================
# [STEP 4] 중앙 정렬 스타일
# =========================================================
st.markdown(
    """
    <style>
      .home-wrap{
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        margin-top:10px;
      }
      .home-title{
        font-size:40px;
        font-weight:800;
        text-align:center;
        margin-bottom:10px;
      }
      .home-sub{
        font-size:18px;
        opacity:0.8;
        text-align:center;
        margin-bottom:22px;
      }
      .home-img{
        display:flex;
        justify-content:center;
        width:100%;
        margin-top:10px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# [STEP 5] 타이틀/서브타이틀 중앙 정렬
# =========================================================
st.markdown(
    """
    <div class="home-wrap">
      <div class="home-title">Mr. Kim… I want to code.</div>
      <div class="home-sub">데이터 기반 고객 이탈 예측 & 마케팅 전략 시뮬레이터</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# [STEP 7] 안내 문구
# =========================================================
st.markdown(
    """
    <div style="text-align:center; margin-top: -5px; opacity: 0.85;">
      상단 메뉴를 통해 <b>세션 분석 · 이탈 예측 · 마케팅 액션</b>을 확인하세요.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# [STEP 6] 대문 이미지 출력 (중앙)
# =========================================================
if img_bytes:
    st.markdown('<div class="home-img">', unsafe_allow_html=True)
    st.image(img_bytes, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ 홈 이미지를 불러오지 못했습니다. Google Drive URL(uc?id=...)과 공유 설정을 확인하세요.")
