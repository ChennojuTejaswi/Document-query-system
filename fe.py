import streamlit as st
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import hashlib
import sqlite3

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Connect to the SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create the users table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

def register_user():
    st.subheader("Register")
    username = st.text_input("Username", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

    if st.button("Register", key="register_button"):
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            try:
                # Insert the new user into the database
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                st.success("Registration successful!")
                st.session_state["show_warning"] = False
            except sqlite3.IntegrityError:
                st.error("Username already exists")

def login_user():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check if the username and password match
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
        user = c.fetchone()

        if user:
            st.session_state["username"] = username
            st.session_state["show_warning"] = False
        else:
            st.error("Invalid username or password")

def get_text_from_docx(docx_files):
    text = ""
    for docx in docx_files:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization =True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Document Query System💁", layout="wide")

    if "username" not in st.session_state:
        st.title("Chat with Word Documents using Gemini")
        register_button = st.button("Register")
        login_button = st.button("Login")

        if register_button:
            register_user()
        elif login_button:
            login_user()
        else:
            st.warning("Please register or log in to access the application.")

    else:
        st.header(f"Welcome, {st.session_state['username']}")
        st.header("Document Query System💁")
        user_question = st.text_input("Ask a Question from the Word Documents")
        if user_question:
            user_input(user_question)
        with st.sidebar:
            st.title("Menu:")
            docx_files = st.file_uploader("Upload your Word Documents and Click on the Submit & Process Button", accept_multiple_files=True, type=["docx"])
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_text_from_docx(docx_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            # Logout button
            logout_button = st.button("Logout")
            if logout_button:
                del st.session_state["username"]
                st.experimental_rerun()  # Clear session state and rerun the app

if __name__ == "__main__":
    main()