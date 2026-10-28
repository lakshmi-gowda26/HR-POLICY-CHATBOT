
import streamlit as st

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG AI — Chat with Your Documents")
st.write("Upload a document and ask questions about its content.")

uploaded_file = st.file_uploader(
    "Upload a text document",
    type=["txt"]
)

if uploaded_file:
    document_text = uploaded_file.getvalue().decode(
        "utf-8", errors="replace"
    )
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.expander("Preview document"):
        st.text(document_text[:3000])

    question = st.text_input("Ask a question about the document")

    if st.button("Search Document"):
        if question.strip():
            # Simple baseline retrieval, not full RAG yet
            chunks = [
                part.strip()
                for part in document_text.split("\n")
                if part.strip()
            ]

            keywords = set(question.lower().split())
            scored = []

            for chunk in chunks:
                score = sum(
                    word in chunk.lower()
                    for word in keywords
                )
                if score > 0:
                    scored.append((score, chunk))

            scored.sort(reverse=True)

            if scored:
                st.subheader("Relevant text")
                for score, chunk in scored[:3]:
                    st.write(chunk)
            else:
                st.warning("No matching text found.")
        else:
            st.warning("Please enter a question.")

st.caption(
    "Prototype: keyword search only. "
    "Embeddings, FAISS, and generated answers will be added next."
)
