from datasets import load_dataset
from langchain.schema import Document
import streamlit as st

@st.cache_resource(show_spinner=True)
def _load_dataset():
    dataset = load_dataset("namphan1999/data-luat", split="train")
    return dataset


@st.cache_resource(show_spinner=True)
def get_allchunks_from_knowledges():
    dataset = _load_dataset()
    allchunks = []
    for idx, ex in enumerate(dataset):
        text = f"{ex['terms']}: {ex['answer']}"
        metadata = {
            "id": idx,
        }
        allchunks.append(Document(page_content=text, metadata=metadata))

    return allchunks