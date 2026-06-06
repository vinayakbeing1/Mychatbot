import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()
loader = PyPDFLoader("22631 WD-APC-UK-HOME(1).pdf")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)    

from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = FAISS.from_documents(chunks, embeddings)


retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature = 0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff")


st.set_page_config(page_title="MyChatbot", page_icon="🤖")
st.title("MyChatbot")

user_question = st.chat_input("Ask something about your policy...")

if user_question:
    st.chat_message("user").write(user_question)
    response = qa_chain.invoke({"query": user_question})
    st.chat_message("assistant").write(response["result"])

