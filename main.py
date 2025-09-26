import streamlit as st
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from models.embedder import get_embedding_model
from retriever.retriever import get_hybrid_retriever
from rerank.reranker import get_rerank_from_retriever
from models.llm import get_answer_from_llm, get_llm
from chunks_data.load_data import get_allchunks_from_knowledges

from cache_communicate.caching_communicate import (
    retrieve_from_cache, 
    store_response, 
    purge_cache
)


# ----------------------------
# Setup Page
# ----------------------------
def setup_page():
    st.set_page_config(
        page_title="ChatBot Pháp Luật Việt Nam",
        page_icon="⚖️",
        layout="wide"
    )


# ----------------------------
# Sidebar Config
# ----------------------------
def setup_sidebar():
    st.sidebar.header("Chức Năng")
    retriver_top_k = 10  # st.sidebar.slider("Số tài liệu tối đa muốn truy vấn (top_k_retriver)", min_value=3, max_value=10, value=10, step=1)
    rerank_top_k = 3     # st.sidebar.slider("Số tài liệu sau rerank", min_value=1, max_value=3, value=3, step=1)

    if st.sidebar.button("🧹 Xóa bộ nhớ đệm"):
        purge_cache()
        st.sidebar.success("Đã xóa cache.")


    st.sidebar.markdown("---")
    st.sidebar.info("Dữ liệu được tải sẵn từ bộ `namphan1999/data-luat`.")

    return retriver_top_k, rerank_top_k


# ----------------------------
# Initialize app (load, embed, retriever, rerank, llm)
# ----------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_system(retriver_top_k: int, rerank_top_k: int):
    # 1) Load knowledge (mỗi phần tử là một chunk tài liệu)
    allchunks = get_allchunks_from_knowledges()

    # 2) Embedding + Vector store
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(allchunks, embeddings)

    # 3) Hybrid retriever (FAISS + BM25)
    retriever_obj = get_hybrid_retriever(vector_store, allchunks, top_k=retriver_top_k, weights=(0.6, 0.8))

    # 4) Reranker
    rerank_obj = get_rerank_from_retriever(retriever_obj, top_k=rerank_top_k)

    return allchunks, retriever_obj, rerank_obj


# ----------------------------
# Chat UI
# ----------------------------
def setup_chat_interface(model_choice):
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("⚖️ Trợ lý Pháp Luật Việt Nam")
    with col2:
        if st.button("🔄 Làm mới hội thoại"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Xin chào! Tôi có thể hỗ trợ bạn về pháp luật Việt Nam."}
            ]
            st.session_state.input_enabled = True
            st.rerun()

    st.caption(f"🚀 Trợ lý AI được hỗ trợ bởi {model_choice}")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể hỗ trợ bạn về pháp luật Việt Nam."}
        ]
        msgs.add_ai_message("Xin chào! Tôi có thể hỗ trợ bạn về pháp luật Việt Nam.")

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs


# ----------------------------
# Handle user input
# ----------------------------
def handle_user_input(msgs, retriever_obj, rerank_obj, model):
    
    if input_user := st.chat_input("Nhập câu hỏi của bạn về pháp luật Việt Nam..."):

        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": input_user})
        st.chat_message("human").write(input_user)
        msgs.add_user_message(input_user)

        # Lấy lịch sử chat gần nhất để làm ngữ cảnh hội thoại
        max_history_pairs = 3
        history_messages = st.session_state.messages[-(max_history_pairs * 2 + 1):-1]
        chat_history_str = "\n".join(
            f"{'Người dùng' if m['role'] == 'human' else 'Bot'}: {m['content']}"
            for m in history_messages
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # 1) Kiểm tra cache
            cached_result = retrieve_from_cache(input_user)
            if cached_result:
                # Hiển thị câu trả lời từ cache
                cached_answer = cached_result.get("response", "")
                placeholder.markdown(cached_answer)
                streamed_text = cached_answer
                st.info("💾 Đang sử dụng câu trả lời từ cache")
            else:
                # 2) Gọi LLM (có streaming)
                try:
                    streamed_text = ""
                    for token in get_answer_from_llm(chat_history_str, input_user, retriever_obj, rerank_obj, model):
                        streamed_text += token
                        placeholder.markdown(streamed_text)
                    answer = streamed_text  # đảm bảo có biến answer để lưu cache
                except Exception as e:
                    answer = f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {e}"
                    placeholder.markdown(answer)

                # 3) Lưu cache
                extra_info = {
                    "model": str(model),
                    "chat_history_length": len(history_messages),
                }
                store_response(input_user, answer, extra_info)
                st.info("🆕 Câu trả lời mới đã được lưu vào cache")

            # Lưu vào session
            st.session_state.messages.append({"role": "assistant", "content": streamed_text})
            msgs.add_ai_message(streamed_text)

# ----------------------------
# Main
# ----------------------------
def main():
    setup_page()
    retriver_top_k, rerank_top_k = setup_sidebar()

    # Khởi tạo hệ thống (dữ liệu, retriever, rerank)
    _, retriever_obj, rerank_obj = bootstrap_system(retriver_top_k, rerank_top_k)

    # Khởi tạo model LLM
    model_LLM = "incept5/llama3.1-claude:latest"
    model = get_llm(model_LLM)

    # Giao diện chat
    msgs = setup_chat_interface(model_LLM)
    handle_user_input(msgs, retriever_obj, rerank_obj, model)


if __name__ == "__main__":
    main()
