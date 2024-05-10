import streamlit as st
from streamlit_chat import message
import tempfile
import os
# import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS

# Create a Streamlit app
st.set_page_config(
    page_title="Chat with CSV",
    page_icon="icons8-alien-16.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title(" ")
st.markdown("<h3 style='text-align: center;'>CSV AI Data ChatBot</h3", unsafe_allow_html=True)
st.title(" ")


# Add an introductory paragraph
st.markdown("""
**Go beyond rows and columns.**  Transform your CSV (Comma Seperated Values) data into a dynamic knowledge base.  Ask questions in plain English and receive insightful answers powered by AI.
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
    
    google_key = os.getenv('GOOGLE_API_KEY')
    # pc_key = os.getenv('PINECONE_API_KEY')
    if not google_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please make sure it's set correctly.")
    # if not pc_key:
    #     st.error("PINECONE_API_KEY not found in .env file. Please make sure it's set correctly.")
    # st.write("GOOGLE_API_KEY:", google_key) 
    # st.write("PINECONE_API_KEY:", pc_key)

    # Embeddings allow transforming the parts cut by CSVLoader into vectors, which then represent an index based on the content of each row of the given file.
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

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Greetings! 😊"]

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm your data assistant! 😊 Ask me anything about " + uploaded_file.name]
        
    response_container = st.container() #container for the chat history
    container = st.container() # container for the user's text input


    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Enter a Prompt here: ", placeholder="What insights are you curious about? Ask here...", key="input")
            submit_form = st.form_submit_button(label='Submit')

            if submit_form and user_input:
                output = conversational_chat(user_input)
            
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="icons")

st.title(" ")
st.title(" ")
st.markdown("""
    <footer style="display: flex; justify-content: center; align-items: center; text-align: center; padding: 15px; border-radius: 10px; margin-top: 20px; box-shadow: 2px 1px 4px rgba(188, 192, 198, 0.38)">
        <p style="font-size: 16px; color: #f0f0f0;">App Developed by 
            <a href="https://twitter.com/raqibcodes" target="_blank" style="color: #90caf9; text-decoration: none; font-weight: bold;">raqibcodes</a>
        </p>
    </footer>
""",
 unsafe_allow_html=True)
    
