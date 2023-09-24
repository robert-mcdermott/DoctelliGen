import os
import sys
import base64

from dotenv import load_dotenv
from glob import glob
import streamlit as st
# following two imports needed for the if __main__ section to start ST from this script
from streamlit.web import cli as stcli  # used in if __main__
from streamlit import runtime # used in if __main__
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"

def init_page():
    st.set_page_config(
        page_title="Document Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    header_html = """
        <style>   
            #watermark-logo {{
                opacity: 0.9;
                transform: scale(0.5);
              }}
        </style>
    <div align="center">
    <img id='watermark-logo' src='data:image/png;base64,{}' class='img-fluid'>
    </div>
    """.format(img_to_bytes("media/logo.png"))

    st.markdown(header_html, unsafe_allow_html=True)
    #st.sidebar.title("Nav")
    if 'costs' not in st.session_state:
        st.session_state.costs = []
    
    # This will hide the streamlit menu and "made with streamlit" footer
    #hide_streamlit_style = """
    #        <style>
    #        [data-testid="stToolbar"] {visibility: hidden !important;}
    #        footer {visibility: hidden !important;}
    #        </style>
    #        """
    #st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
def select_model():
    #model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    #if model == "GPT-3.5":
    #    st.session_state.model_name = "gpt-3.5-turbo"
    #elif model == "GPT-3.5":
    #    st.session_state.model_name = "gpt-3.5-turbo-16k"
    #else:
    #    st.session_state.model_name = "gpt-4"

    st.session_state.model_name = "gpt-3.5-turbo-16k"
    temperature = st.sidebar.slider('Temperature', 0.0, 2.0, 0.3, 0.1)
    # 300: The number of tokens for instructions outside the main text
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=temperature, model_name=st.session_state.model_name)


def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here 📃',
        type='pdf'
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            # The appropriate chunk size needs to be adjusted based on the PDF being queried.
            # If it's too large, it may not be able to reference information from various parts during question answering.
            # On the other hand, if it's too small, one chunk may not contain enough contextual information.
            chunk_size=500,
            chunk_overlap=100,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # Get all collection names.
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # If the collection does not exist, create it.
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)
    # You can also do it like this. In this case, the vector database will be initialized every time.
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )


def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # There are also "mmr," "similarity_score_threshold," and others.
        search_type="similarity",
        # How many documents to retrieve? (default: 4)
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def ask(qa, query):
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)
    return answer, cb.total_cost


def page_ask_my_pdf():
    st.title("Document Q&A")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Enter Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("The LLM is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


# so we can display images in streamlit
def img_to_bytes(img):
    with open(img, "rb") as f:
        img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    init_page()

    selection = st.sidebar.radio("Go to", ["Ask Questions", "PDF Upload"])
    if selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Ask Questions":
        page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


if __name__ == '__main__':
    load_dotenv()
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())