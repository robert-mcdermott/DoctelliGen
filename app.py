import os
import sys
import base64
from io import StringIO

from dotenv import load_dotenv
from glob import glob
import streamlit as st
# following two imports needed for the if __main__ section to start ST from this script
from streamlit.web import cli as stcli  # used in if __main__
from streamlit import runtime # used in if __main__
from streamlit_option_menu import option_menu
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
COLLECTION_NAME = "collection01"

def init_page():
    """
    Initialize the webapp
    """
    st.set_page_config(
        page_title="Document Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
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
    """
    Select the OpenAI model to use (turbo-16k for largest context,
    set the max context context size,
    set the temperature (0.3 seems to work well),
    and return the LLM object
    """
    st.session_state.model_name = "gpt-3.5-turbo-16k"
    temperature = 0.3
    # 300: The number of tokens for instructions outside the main text
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=temperature, model_name=st.session_state.model_name)

def get_text_embeddings():
    """
    Extract text from the documents, chunk it up with some overlap, convert to embeddings and return chunks.
    you can set the embedding model, chunk size and overlap amount here
    """
    uploaded_file = st.file_uploader(
        label='Upload your documents here',
        type=['txt','pdf','md']
    )
    if uploaded_file:
        if uploaded_file.name.endswith('.pdf'.lower()):
            pdf_reader = PdfReader(uploaded_file)
            text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        else:
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).getvalue()

        #The chunk size and overlap might need to be adjusted based on document size
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            chunk_size=500,
            chunk_overlap=100,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    """
    Create a quadrant client, open the DB and collection and return it
    """
    client = QdrantClient(path=QDRANT_PATH)
    # Get all collection names.
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    for collection in collection_names:
        print("collection: {}".format(collection))

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

def build_vector_store(text):
    """
    Write the embeddings to the vector DB
    """
    qdrant = load_qdrant()
    qdrant.add_texts(text)
    # You can also do it like this. In this case, the vector database will be initialized every time.
    # Qdrant.from_texts(
    #     text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name="my_documents",
    # )


def build_qa_model(llm):
    """
    Build the embedding query object that will be used to find chunks to return 
    """
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # There are also "mmr," "similarity_score_threshold," and others.
        search_type="similarity",
        # How many documents to retrieve? This should be adjusted to fit in your context size
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

def page_doc_upload():
    """
    Page where documents are uploaded
    """
    st.title("📄Document Upload ⬆️")
    container = st.container()
    with container:
        text = get_text_embeddings()
        if text:
            with st.spinner("Loading Document ..."):
                build_vector_store(text)

def ask(qa, query):
    """
    Send question to LLM
    """
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)
    return answer, cb.total_cost

def page_ask_docs():
    """
    Page to answer questions
    """
    st.title("📄Document Q&A❓")
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
                with st.spinner("🤖 The LLM is typing..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None
        if answer:
            with response_container:
                st.markdown("## Answer")
                st.markdown(answer['result'])

# so we can display images in streamlit
def img_to_bytes(img):
    """
    Fetch and image from from the disk and return it as bytes so streamlit can display it
    """
    with open(img, "rb") as f:
        img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    init_page()
    #This gets and prints the OpenAI charges. commented out as not somethings we'd want to show users
    costs = st.session_state.get('costs', [])
    #st.sidebar.markdown("## Costs")
    #st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    #for cost in costs:
    #    st.sidebar.markdown(f"- ${cost:.5f}")
    with st.sidebar:
        selection2 = option_menu(
            menu_title='Tasks',
            options=['Ask Questions','Upload Docs'],
            icons=['bi-chat-left-dots-fill','cloud-upload-fill'],
            menu_icon='bi-list',
            default_index=0,
            styles={
                "container": {"padding": "1!important","background-color":'#0A799A'},
                "menu-title": {"color": "white", "font-size": "20px", "font-weight": "bold"},
                "icon": {"color": "white", "font-size": "16px"}, 
                "nav-link": {"color":"white","font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#FFB500"},
                "nav-link-selected": {"background-color": "#AA4AC4"},}    
            )
    if selection2 == "Ask Questions":
        page_ask_docs()
    if selection2 == "Upload Docs":
        page_doc_upload()

if __name__ == '__main__':
    load_dotenv()
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())