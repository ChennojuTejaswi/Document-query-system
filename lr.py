import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sqlite3
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import hashlib
from docx import Document


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not available in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def save_embeddings(embeddings):
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    for embedding in embeddings:
        cursor.execute("INSERT INTO embeddings (embedding) VALUES (?)", (embedding,))
    conn.commit()
    conn.close()

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except TimeoutError:
        st.error("Error: Timed out while embedding your question. Please try again.")
        return

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def register_user(username, password):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Ensure 'hashed_password' column exists before inserting
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password BLOB')
    except sqlite3.OperationalError:
        pass  # Column already exists, ignore the error
    cursor.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

def login_user(username, password):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user is not None  # Check if
def main():
    # Create database connections (if they don't exist)
    conn_embeddings = sqlite3.connect('embeddings.db')
    cursor_embeddings = conn_embeddings.cursor()
    cursor_embeddings.execute('''CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, embedding BLOB)''')
    conn_embeddings.commit()

    conn_users = sqlite3.connect('users.db')
    cursor_users = conn_users.cursor()

    # Create the 'users' table with 'hashed_password' column
    cursor_users.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, hashed_password BLOB)''')
    conn_users.commit()

    hashed_password = hashlib.sha256("lucky684".encode('utf-8')).hexdigest()  # Replace with a secure way to get password

    st.set_page_config(page_title="Chat with Documents", layout="wide")

    # Login logic
    if "username" not in st.session_state:
        st.title("Chat with Word Documents using Gemini")

        # Registration form
        st.subheader("Register")
        new_username = st.text_input("Username:", key="username_input")  # Unique key for username
        new_password = st.text_input("Password:", type="password", key="password_input")  # Unique key for password
        register_button = st.button("Register")

        if register_button:
            if new_username and new_password:
                if not login_user(new_username, new_password):  # Check for existing user with same username
                    register_user(new_username, new_password)
                    st.success("Registration Successful!")
                else:
                    st.error("Username already exists")
            else:
                st.error("Please fill both username and password")

        st.subheader("Login")
        username = st.text_input("Username:", key="login_username")  # Unique key for login username
        password = st.text_input("Password:", type="password", key="login_password")  # Unique key for login password
        login_clicked = st.button("Login")

        if login_clicked:
            if login_user(username, password):
                st.session_state["username"] = username
            else:
                st.error("Invalid username or password")

    else:
        st.header("Chat with Word Documents using Gemini")
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
                    # Save embeddings to database
                    embeddings = [chunk.embedding for chunk in text_chunks]
                    save_embeddings(embeddings)
                    st.success("Done")
        # Logout button
        logout_button = st.button("Logout")
        if logout_button:
            del st.session_state["username"]
            st.experimental_rerun()  # Clear session state and rerun the app

    # Close database connections
    conn_embeddings.close()
    conn_users.close()

if __name__ == "__main__":
    main()

