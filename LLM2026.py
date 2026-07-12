from dotenv import load_dotenv
load_dotenv()
import os
import base64
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="HarshaDeepJoga-Multimodal AI",
    page_icon="🤖",
    layout="wide"
)
st.header("My Multimodal AI Assitant")
st.caption("📄 Chat with PDFs • 🖼 Analyze Images • ⚡ Powered by Groq, LangChain & FAISS")

st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#FFF9C4;
}

/* Buttons */
.stButton>button{
    background-color:#FF8C00;
    color:white;
    border-radius:8px;
    border:none;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    border:2px solid #FF8C00;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if len(chunks) == 0:
        st.error("PDF produced zero text.")
        st.stop()

    # Embeddings (free)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    # Vector DB
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User input
    user_question = st.text_input("Type your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)

        # Groq LLM
        LLM = ChatGroq(
            groq_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama-3.3-70b-versatile",   
            temperature=0,
            max_tokens=1000
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n{context}\nQuestion: {question}"
        )

        # Chain
        chain = (
            {
                "context": lambda x: "\n\n".join(
                    doc.page_content for doc in x["input_documents"]
                ),
                "question": RunnablePassthrough()
            }
            | prompt
            | LLM
        )

        # Response
        response = chain.invoke({
            "input_documents": match,
            "question": user_question
        })

        st.write(response.content)


with st.sidebar:
    st.title("Image")
    img = st.file_uploader("upload image",type=["jpg","jpeg","png"])

if img is not None:
    image = Image.open(img)
    st.image(image)

    bas64img = base64.b64encode(img.getvalue()).decode("utf-8")

    ImageLM = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.15
    )

    # User input
    user_question1 = st.text_input("Type your question here")

    if user_question1:

        message = HumanMessage(
        content=[
            {"type": "text", "text":f"""You are an expert vision assistant. Analyze the uploaded image carefully and answer the following question,
            Question:
            {user_question1}"""},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{bas64img}"
                },
            },
        ]
    )
        result = ImageLM.invoke([message])
        st.write(result.content)
