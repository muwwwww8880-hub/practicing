import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError, PermissionDenied
from google.generativeai.types import StopCandidateException, BlockedPromptException
import pandas as pd
import io
import time
from datetime import datetime
import os

# --- 시스템 프롬프트 (System Prompt) ---
SYSTEM_PROMPT = """
1) 당신은 사용자와 함께 이야기를 만들어가는 '이야기 작가 봇'입니다. 항상 친절하고 명랑한 어투를 유지해주세요.
2) 사용자가 만들고 싶은 이야기의 주제를 물어보세요. 만약 사용자가 주제를 추천해달라고 한다면, 흥미로운 주제를 1~2가지 제안해주세요.
3) 주제가 정해지면, 당신이 먼저 이야기의 '발단' 부분(시작)을 2-3줄 정도로 제시합니다. 그리고 사용자에게 그 뒷이야기를 이어서 작성해달라고 요청하세요.
4) 사용자가 응답하면, 당신은 그 이야기에 이어지는 '전개' 또는 '위기' 부분 2-3줄을 작성합니다. 이 과정을 이야기가 완성될 때까지 반복합니다.
5) 이야기의 구성(발단-전개-위기-절정-결말)을 고려하며, 사용자의 이야기에 자연스럽게 이어지는 내용을 작성해야 합니다.
6) 사용자가 '이야기를 마무리해줘'라고 요청하거나, 내용상 '결말'에 도달했다고 판단되면, 이야기 작성을 마무리하는 결론을 작성하세요.
7) 이야기가 완성되면, "와, 멋진 이야기가 완성되었네요! 짝짝짝!" 같은 칭찬의 말과 함께,
   지금까지 주고받은 '사용자'와 '당신'의 모든 이야기 조각들을 하나의 완성된 글로 합쳐주세요.
   그리고 그 완성된 이야기에 어울리는 '임의의 제목'을 1개 붙여서 사용자에게 전달해주세요.
   마지막으로, 완성된 이야기에 대해 '한 줄 평가'를 남겨주세요.
"""

# --- 1. 앱 설정 및 초기화 ---

st.set_page_config(
    page_title="AI 이야기 이어쓰기 챗봇",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("✍️ AI 이야기 이어쓰기 챗봇")
st.write("Gemini AI와 함께 새로운 아이디어를 탐험하고 멋진 이야기를 완성해보세요!")

# 기본 모델 정의
DEFAULT_MODEL = "models/gemini-2.0-flash"
FALLBACK_MODELS = [DEFAULT_MODEL, "models/gemini-pro"]

@st.cache_data(ttl=3600)
def get_available_models(api_key):
    """Gemini API에서 사용 가능한 모델 목록을 가져옵니다. (exp 제외)"""
    try:
        genai.configure(api_key=api_key)
        model_list = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods and \
               'exp' not in model.name:
                model_list.append(model.name)
        
        if DEFAULT_MODEL in model_list:
            model_list.remove(DEFAULT_MODEL)
            model_list.insert(0, DEFAULT_MODEL)
        elif "models/gemini-pro" in model_list:
             pro_model = "models/gemini-pro"
             model_list.remove(pro_model)
             model_list.insert(0, pro_model)
        
        return model_list if model_list else FALLBACK_MODELS
    
    except PermissionDenied as e:
        st.error(f"API 권한 오류: {e}. Google Cloud에서 Gemini API가 활성화되었는지 확인하세요.")
        return FALLBACK_MODELS
    except Exception as e:
        st.error(f"모델 목록 로딩 실패 (API 키가 유효한가요?): {e}")
        return FALLBACK_MODELS

def init_api_key():
    """API 키를 st.secrets 또는 사용자 입력으로 받습니다. (오류 방지 강화)"""
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass 

    if api_key:
        return api_key
    
    st.sidebar.warning("API 키가 설정되지 않았습니다. st.secrets에 'GEMINI_API_KEY'를 추가해주세요.")
    api_key = st.sidebar.text_input("또는 여기에 Gemini API 키를 임시 입력하세요:", type="password")
    
    if not api_key:
        st.error("Gemini API 키가 필요합니다. 사이드바에서 입력해주세요.")
        st.stop()
        
    return api_key

# --- 2. 세션 상태 관리 ---

def init_session_state():
    """Streamlit 세션 상태를 초기화합니다."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "log_data" not in st.session_state:
        st.session_state.log_data = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL

def start_new_chat_session(model_name, history=None):
    """새로운 챗 세션을 시작하거나, 기존 기록으로 세션을 복원합니다."""
    try:
        model = genai.GenerativeModel(
            model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config={"temperature": 0.8}
        )
        
        formatted_history = []
        if history:
            for msg in history:
                formatted_history.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [msg["content"]]
                })
        
        st.session_state.chat_session = model.start_chat(history=formatted_history)
        st.session_state.selected_model = model_name
        
    except Exception as e:
        st.error(f"챗 세션 시작에 실패했습니다: {e}")
        st.stop()

def clear_chat_history():
    """모든 대화 기록과 세션 정보를 초기화합니다."""
    st.session_state.messages = []
    st.session_state.log_data = []
    st.session_state.chat_session = None
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # st.session_state.selected_model은 유지합니다.
    st.rerun()

def prune_and_restart_session(keep_messages=6):
    """API 한도(429) 도달 시, 최근 N개의 메시지만 남기고 세션을 다시 시작합니다."""
    st.warning(f"API 한도(429) 도달. 최근 {keep_messages}개(3턴)의 대화만 남기고 세션을 재시작합니다.")
    time.sleep(2) 

    st.session_state.messages = st.session_state.messages[-keep_messages:]
    st.session_state.log_data = st.session_state.log_data[-keep_messages:]

    start_new_chat_session(
        st.session_state.selected_model,
        history=st.session_state.messages
    )

# --- 3. UI 및 상호작용 (사이드바) ---

# 세션 상태가 먼저 초기화되었는지 확인
init_session_state()

# API 키 가져오기
API_KEY = init_api_key()
# API 키로 모델 목록 가져오기
MODELS = get_available_models(API_KEY) 

# --- [수정된 부분] Selectbox를 위한 안전한 인덱스 계산 ---
# MODELS가 비어있거나(None) 기본 모델이 목록에 없는 비상 상황 대비
model_list = MODELS if (MODELS and isinstance(MODELS, list)) else FALLBACK_MODELS

# 현재 세션의 모델이 목록에 있는지 확인
if st.session_state.selected_model not in model_list:
    st.session_state.selected_model = model_list[0] # 없으면 목록의 첫 번째 모델로 강제 설정

# 인덱스 계산 (ValueError 방지)
try:
    current_index = model_list.index(st.session_state.selected_model)
except ValueError:
    current_index = 0
# --- [수정 완료] ---


st.sidebar.title("설정")

# 모델 선택 (안전하게 수정된 인덱스 사용)
selected_model_name = st.sidebar.selectbox(
    "AI 모델 선택",
    options=model_list,
    index=current_index, # 여기!
    help="이야기 만들기에 사용할 Gemini 모델을 선택합니다."
)

# 모델이 변경되었으면 세션 초기화
if selected_model_name != st.session_state.selected_model:
    clear_chat_history()
    # 새 모델로 세션 시작 (clear_chat_history가 rerun하므로 다음 실행에서 적용됨)
    # start_new_chat_session(selected_model_name, history=None) # 여기서 호출하면 안 됨

# 새 이야기 시작 버튼
st.sidebar.button("새 이야기 시작", on_click=clear_chat_history, use_container_width=True)

# CSV 기록 옵션
log_enabled = st.sidebar.checkbox("대화 자동 기록 (CSV)", value=True, help="체크하면 모든 대화가 CSV 다운로드를 위해 임시 저장됩니다.")

# 로그 다운로드
def get_log_csv():
    if not st.session_state.log_data:
        return ""
    log_df = pd.DataFrame(st.session_state.log_data)
    return log_df.to_csv(index=False).encode('utf-8-sig')

st.sidebar.download_button(
    label="대화 기록 다운로드 (CSV)",
    data=get_log_csv(),
    file_name=f"chat_log_{st.session_state.session_id}.csv",
    mime="text/csv",
    use_container_width=True,
    disabled=not st.session_state.log_data
)

# 세션 정보 표시
st.sidebar.markdown("---")
st.sidebar.subheader("세션 정보")
st.sidebar.info(f"""
**Model:** `{st.session_state.selected_model.split('/')[-1]}`
**Session ID:** `{st.session_state.session_id}`
""")

# --- 4. 메인 챗 로직 ---

# 챗 세션이 없는 경우 (첫 실행 또는 초기화 후) 시작
if st.session_state.chat_session is None:
    start_new_chat_session(st.session_state.selected_model, history=None)

# 이전 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("여기에 이야기나 아이디어를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if log_enabled:
        st.session_state.log_data.append({
            "timestamp": datetime.now(),
            "session_id": st.session_state.session_id,
            "model": st.session_state.selected_model,
            "role": "user",
            "content": prompt
        })
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("AI가 이야기를 잇고 있습니다..."):
            try:
                response = st.session_state.chat_session.send_message(prompt)
                bot_response = response.text
                
            except ResourceExhausted as e:
                prune_and_restart_session(keep_messages=6)
                try:
                    response = st.session_state.chat_session.send_message(prompt)
                    bot_response = response.text
                except Exception as e2:
                    st.error(f"재시도 실패: {e2}")
                    bot_response = None
            
            except (StopCandidateException, BlockedPromptException) as e:
                st.error("⚠️ 안전 설정에 의해 응답이 차단되었습니다. 다른 주제로 이야기해 주세요.")
                bot_response = None
            
            except InternalServerError as e:
                st.error("🔌 현재 Gemini API 서버에 문제가 발생했습니다. (500) 잠시 후 다시 시도해주세요.")
                bot_response = None

            except Exception as e:
                st.error(f"예상치 못한 오류가 발생했습니다: {e}")
                bot_response = None

            if bot_response:
                message_placeholder.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                if log_enabled:
                    st.session_state.log_data.append({
                        "timestamp": datetime.now(),
                        "session_id": st.session_state.session_id,
                        "model": st.session_state.selected_model,
                        "role": "assistant",
                        "content": bot_response
                    })