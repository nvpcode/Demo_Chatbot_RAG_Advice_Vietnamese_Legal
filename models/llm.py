from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retriever.retriever import show_retrieved_chunks
from rerank.reranker import show_rerank_chunks

def get_llm(model_choice):
    """
    Trả về LLM object của LangChain
    """
    return ChatOllama(
        model=model_choice,
        temperature=0.001,
        base_url="http://localhost:11434",
        device="cuda"  # Chạy trên gpu
    )

def get_answer_from_llm(chat_history, user_question, retriever_obj, rerank_obj, model):
    # Lấy các tài liệu liên quan
    retrieved_docs = show_retrieved_chunks(retriever_obj, user_question)
    reranked_docs = show_rerank_chunks(rerank_obj, user_question)
    context = "\n".join(doc.page_content for doc in reranked_docs)
    print("📏 Context length:", len(context), "ký tự")
    print(f"Context for LLM:\n{context}\n{'-'*50}\n")

    # Tạo ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """"Bạn là một chuyên gia am hiểu luật pháp Việt Nam. Trả lời bằng Tiếng Việt.
         Chỉ trả lời dựa trên thông tin có trong tài liệu và lịch sử hội thoại đã được cung cấp bên dưới.
         Tuyệt đối KHÔNG được suy đoán, bịa đặt hoặc đưa thông tin ngoài phạm vi tài liệu.
         Hãy trả lời rõ ràng, chính xác, trích dẫn điều luật nếu có thể.

        --- Lịch sử hội thoại ---
        {chat_history}

        --- Tài liệu hỗ trợ ---
        {context}
        """
        ),
        ("human", "{question}")
    ])

    # Format prompt với dữ liệu thực tế
    messages = prompt.format_messages(
        chat_history=chat_history,
        context=context,
        question=user_question
    )

    # Streaming từng chunk nội dung
    for chunk in model.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content