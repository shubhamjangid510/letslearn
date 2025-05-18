import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
from datetime import datetime
import time

from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()


# Database configuration
DB_CONFIG = {
    "dbname": "class24_chatbot",
    "user": "class24_chatbot_user",
    "password": "0jb4pWK4vXtZR5H8jyxSY7YWWhDhypga",
    "host": "dpg-d07smm2dbo4c73bs56fg-a.singapore-postgres.render.com",
    "port": "5432"
}

# Set page config
st.set_page_config(page_title="RAG Application", layout="wide")


# Database connection
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'File Upload'
if 'user_id' not in st.session_state:
    st.session_state.user_id = '1'  # Hardcoded user_id
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_key' not in st.session_state:
    st.session_state.input_key = str(uuid.uuid4())

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def create_chat_session(user_id, title="New Chat"):
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, user_id, title, datetime.now())
            )
        conn.commit()
    finally:
        conn.close()
    return session_id

def update_chat_session_title(session_id, new_title):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions 
                SET title = %s 
                WHERE id = %s
                """,
                (new_title[:200], session_id)
            )
        conn.commit()
    finally:
        conn.close()

def save_chat_history(user_id, session_id, question, answer):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (user_id, session_id, question, answer, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, session_id, question, answer, datetime.now())
            )
        conn.commit()
    finally:
        conn.close()

def get_chat_history(user_id, session_id):
    if not session_id:
        return []
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT question, answer 
                FROM chat_history 
                WHERE user_id = %s AND session_id = %s 
                ORDER BY created_at ASC
                """
            params = (user_id, session_id)
            print("\n\n------------------\n\n", cur.mogrify(query, params).decode("utf-8"))
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()

def get_chat_sessions(user_id):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, title 
                FROM chat_sessions 
                WHERE user_id = %s 
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            return cur.fetchall()
    finally:
        conn.close()

def get_documents():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, name FROM documents")
            return cur.fetchall()
    finally:
        conn.close()

def process_document(file):
    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
    
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (name, created_at) 
                VALUES (%s, %s) 
                RETURNING id
                """,
                (file.name, datetime.now())
            )
            doc_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    
    for split in splits:
        split.metadata["document_id"] = doc_id
    
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    vector_store = PGVector(
        collection_name="rag_embeddings",
        connection_string=connection_string,
        embedding_function=embeddings
    )
    vector_store.add_documents(splits)
    
    os.remove(temp_path)
    return doc_id

# # CSS for chat styling (minimal, professional)
# st.markdown("""
# <style>
# .stChatMessage {
#     margin: 10px 0;
#     padding: 10px;
#     border-radius: 10px;
#     max-width: 80%;
# }
# .stChatMessage[data-testid="stChatMessage-User"] {
#     background-color: #d1e7dd;
#     margin-left: auto;
# }
# .stChatMessage[data-testid="stChatMessage-Assistant"] {
#     background-color: #f8d7da;
# }
# </style>
# """, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Let's Learn!!!")
    page = st.radio("Select Page", ["File Upload", "Chat"])
    st.session_state.page = page
    
    
    if page == "Chat":
        # Create two columns for subheader and button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Chat Sessions")
        
        with col2:
            if st.button('📝', help= "New Chat", key="user_chats"):
                st.session_state.session_id = create_chat_session(st.session_state.user_id)
                st.session_state.chat_history = []
        sessions = get_chat_sessions(st.session_state.user_id)
        MIN_TITLE_LENGTH = 20
        # Session buttons with padded/truncated titles
        for session in sessions:
            # Truncate long titles and pad short ones
            display_title = session['title']
            if len(display_title) > MIN_TITLE_LENGTH:
                display_title = display_title[:MIN_TITLE_LENGTH] + "..."
            elif len(display_title) < MIN_TITLE_LENGTH:
                display_title = display_title + str(" " * (MIN_TITLE_LENGTH - len(display_title)))
            
            # Place each button in a container with a single column
            with st.container():
                col = st.columns([4])[0]
                with col:
                    if st.button(display_title, key=session['id'], type="secondary", help=session['title']):
                        st.session_state.session_id = session['id']
                        db_history = get_chat_history(st.session_state.user_id, st.session_state.session_id)
                        st.session_state.chat_history = [
                            {"role": "user", "content": item["question"]}
                            for item in db_history
                        ] + [
                            {"role": "assistant", "content": item["answer"]}
                            for item in db_history
                        ]

# Page 1: File Upload
def upload_page():
    st.title("Document Upload")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                doc_id = process_document(uploaded_file)
                st.success(f"Document processed successfully! ID: {doc_id}")

# Page 2: Chat Interface
def chat_page():
    st.title("Lets Learn Together!!!")
    
    # Initialize vector store
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    vector_store = PGVector(
        collection_name="rag_embeddings",
        connection_string=connection_string,
        embedding_function=embeddings
    )
    
    # Initialize conversation chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Load chat history for the selected session
    db_history = get_chat_history(st.session_state.user_id, st.session_state.session_id)
    for item in db_history:
        memory.save_context({"input": item['question']}, {"output": item['answer']})
    
    custom_prompt_template = """
        You are LearnBot, a conversational chatbot designed to help users learn by answering questions based on uploaded documents and maintaining a friendly, interactive conversation. Your capabilities include:
        - Answering questions using a knowledge base built from uploaded documents.
        - Engaging in conversational dialogue, leveraging chat history to maintain context and answer follow-up questions.
        - Responding to questions about your capabilities or the conversation itself (e.g., recalling the last question asked).

        Follow these guidelines:
        1. **Document-Based Questions**: For questions related to specific topics or information, answer using only the provided context from uploaded documents and the chat history. Do not use external knowledge or make assumptions beyond what's given.
        2. **Meta-Questions**: For questions about your capabilities (e.g., "What can you do?") or the conversation (e.g., "What was my last question?"), provide accurate answers based on your defined capabilities or by analyzing the chat history.
        3. **Handle Missing Information**: If a document-based question lacks relevant information in the context or chat history, respond with: "I don't have much information for your question, please try another question!"
        4. **Be Clear and Concise**: Provide accurate, straightforward answers in a friendly and engaging tone.
        5. **Contextual Awareness**: Use the chat history to maintain conversation flow, avoid repetition, and answer follow-up questions appropriately.

        Chat History:
        {chat_history}

        Context from Documents:
        {context}

        Question:
        {question}

        Answer:
        """
    custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["chat_history", "context", "question"]
        )
            
    retriever = vector_store.as_retriever()
    print("\n\n-------------------------Memory ->\n\n------------", memory)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question:", key=st.session_state.input_key):
        # Create a new session if none exists
        if not st.session_state.session_id:
            st.session_state.session_id = create_chat_session(st.session_state.user_id, prompt)
        
        # Append and display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Update session title if first question
        if not db_history:
            update_chat_session_title(st.session_state.session_id, prompt)
            
        
        # Generate and stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                result = qa_chain({"question": prompt})
                answer = result['answer']
            # Stream response
            response_placeholder = st.empty()
            streamed_text = ""
            for char in answer:
                streamed_text += char
                response_placeholder.markdown(streamed_text)
                time.sleep(0.01)  # Simulate streaming
            answer = streamed_text.replace("$", "\$")
            
            # Save to history and database
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            save_chat_history(
                st.session_state.user_id,
                st.session_state.session_id,
                prompt,
                answer
            )
        
        # Clear input
        st.session_state.input_key = str(uuid.uuid4())
        # st.rerun()

# Main app
def main():
    if st.session_state.page == 'File Upload':
        upload_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()