from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def get_hybrid_retriever(vector_store, all_chunks_docs, top_k, weights=(0.5, 0.5)): 
    """
    Tạo object retriever Hybrid kết hợp Dense (FAISS) + Sparse (BM25)
    với tỉ lệ trọng số tốt nhất là (0.5, 0.5)
    """

    # Dense (FAISS)
    retriever_faiss = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    # Sparse (BM25)
    retriever_bm25 = BM25Retriever.from_documents(all_chunks_docs)
    retriever_bm25.k = top_k

    retriever_obj = EnsembleRetriever(
        retrievers=[retriever_faiss, retriever_bm25],
        weights=list(weights) 
    )

    return retriever_obj

# Lưu ý: 
# Hàm này trả về object retriever, không phải danh sách document. 
# Muốn nó tìm ra tài liệu liên quan tới input đầu vào thì gọi retriever_obj.invoke(query)

def show_retrieved_chunks(retriever_obj, query: str):
    """
    Hàm nhận retriever_obj và query đầu vào.
    Đầu ra: số lượng chunks và nội dung của từng chunk.
    """
    # Lấy danh sách các tài liệu liên quan
    retrieved_docs = retriever_obj.invoke(query)

    print(f"Top docs after retriever:")

    # # In ra nội dung từng chunk
    # for i, doc in enumerate(retrieved_docs, 1):
    #     print(f"--- Chunk {i} ---")
    #     print(doc.page_content.strip())
    #     if doc.metadata:
    #         print(f"📌 Metadata: {doc.metadata}")
    #     print()

    # In ra số lượng chunks
    print(f"🔎 Số lượng chunks lấy được: {len(retrieved_docs)}")
    
    return retrieved_docs

