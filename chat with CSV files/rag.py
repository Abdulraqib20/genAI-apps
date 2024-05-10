import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import os
from dotenv import load_dotenv
import yaml
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI as Gemini

# Create a Streamlit app
st.set_page_config(
    page_title="Chat with CSV",
    page_icon="icons8-whatsapp-48.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(" ")
st.markdown("<h3 style='text-align: center;'>Generative AI Application to Converse with your CSV files data</h3", unsafe_allow_html=True)
st.title(" ")
# st.image("rachit-tank-lZBs-lD9LPQ-unsplash.jpg")


# Add an introductory paragraph
st.markdown("""
Generative AI Application to Converse with your CSV files data
""")

user_api_key = st.sidebar.text_input(
    label="#### Enter your Gemini API key, here",
    placeholder="XXXXXXXXXX",
    type="password")

uploaded_file = st.sidebar.file_uploader("Upload your CSV (comma seperated values) file:", type="csv")

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    # st.write(data)

    ### Load environment variables from .env file
    load_dotenv()
    google_key = os.getenv('GOOGLE_API_KEY')
    pc_key = os.getenv('PINECONE_API_KEY')
    if not google_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please make sure it's set correctly.")
    if not pc_key:
        st.error("PINECONE_API_KEY not found in .env file. Please make sure it's set correctly.")
    # st.write("GOOGLE_API_KEY:", google_key) 
    # st.write("PINECONE_API_KEY:", pc_key)

    ### Load configuration
    # with open("config.yaml", 'r') as file:
    #     config = yaml.safe_load(file)
    # google_api_key = config['api_keys']['google']
    # pinecone_api_key = config['api_keys']['pinecone']
    # st.write(google_api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
    vectorstore = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm = Gemini(model='models/gemini-pro', temperature=0, google_api_key=google_key
    ),
    retriever=vectorstore.as_retriever())

    def conversational_chat(query):
        result = chain({
            'question': query,
            'chat_history': st.session_state['history']
        })
        st.session_state['history'].append((query, result['answer']))

        return result['answer']
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    response_container = st.container() #container for the chat history
    container = st.container() # container for the user's text input


    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query: ", placeholder="Talk about your csv data here (:", key="input")
            submit_form = st.form_submit_button(label='Send')

            if submit_form and user_input:
                output = conversational_chat(user_input)
            
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
    