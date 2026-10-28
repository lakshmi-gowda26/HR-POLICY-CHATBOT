import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="centered"
)

POLICY = """
Annual Leave:
Employees are entitled to 24 days of annual leave per calendar year.
Leave requests should normally be submitted at least 5 working days in advance.

Sick Leave:
Employees who are unable to work due to illness should inform their manager
before the start of their working day. A medical certificate may be required
for extended periods of absence.

Paternity Leave:
Eligible female employees are entitled to 15 working days of paternity leave.
The leave should be taken within six months of the child's birth.

Work From Home:
Employees may work from home up to 2 days per week with manager approval.
Employees must remain available during normal working hours.

Travel Expenses:
Employees travelling for business can claim reasonable hotel, transportation,
and meal expenses. Original receipts should be submitted within 30 days
of returning from the trip.
"""

# Keep each policy section together
chunks = [
    section.strip()
    for section in POLICY.strip().split("\n\n")
    if section.strip()
]

@st.cache_resource
def load_search_system():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))

    return model, index

# Styling
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    max-width: 800px;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏢 HR Policy Assistant")
st.write(
    "Your company policy helper. Ask about leave, "
    "work from home, or travel expenses."
)

st.info("Ask a question to retrieve relevant HR policy sections.")

# Example questions
examples = [
    "How many days of annual leave can I take?",
    "Can a male employee take paternity leave?",
    "How many days can I work from home?",
    "When should I submit travel receipts?"
]

selected = st.selectbox(
    "Try an example question",
    ["Choose an example..."] + examples
)

question = st.text_input(
    "Your question",
    value="" if selected == "Choose an example..." else selected,
    placeholder="Type your HR question here..."
)

if st.button("🔍 Search HR Policy", type="primary",
             use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching policy documents..."):
            model, index = load_search_system()

            query_embedding = model.encode([question])
            query_vector = np.asarray(
                query_embedding, dtype="float32"
            )

            k = min(3, index.ntotal)
            distances, indices = index.search(query_vector, k)

            results = [
                chunks[i]
                for i in indices[0]
                if i >= 0
            ]

        st.subheader("📄 Retrieved policy sections")

        for number, result in enumerate(results, start=1):
            with st.expander(
                f"Policy result {number}",
                expanded=(number == 1)
            ):
                st.write(result)

        st.subheader("💬 Policy guidance")

        if "paternity" in question.lower():
            st.warning(
                "The supplied policy states that eligible female "
                "employees receive 15 working days of paternity leave, "
                "to be taken within six months of the child's birth. "
                "It does not clarify male employee eligibility. "
                "Please confirm with HR."
            )
        else:
            st.write(
                "Use the retrieved policy sections above to review "
                "the rules relevant to your question. This version "
                "retrieves text but does not yet generate a complete "
                "AI-written answer."
            )

        st.caption(
            "For unclear or missing rules, contact your HR department."
        )

st.divider()
st.caption("HR Policy Assistant | Powered by FAISS and Sentence Transformers")