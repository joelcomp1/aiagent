import glob
import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import  VectorParams, Distance,
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ENV:
    COLLECTION_NAME = "county"
    VECTOR_NAME = "text_embedding"
    CHUNK_SIZE=3072 #matches the OpenAI embedding library


class ChatBot:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key= os.getenv("QDRANT_API_KEY"),
        )

        self.text_embedding_model: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        self.llm = ChatOpenAI()
        self.text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
        )

    def processing_local_data(self):
        qdrant: QdrantVectorStore
        if not self.qdrant_client.collection_exists(
            collection_name=ENV.COLLECTION_NAME
        ):
            os.write(1,f"Need to create data collection: {ENV.COLLECTION_NAME}".encode())
            
            self.qdrant_client.create_collection(
                ENV.COLLECTION_NAME,
                vectors_config=VectorParams(
                        size=ENV.CHUNK_SIZE,  # Dimension of text embeddings
                        distance=Distance.COSINE,  # Cosine similarity is used for comparison
                    )
            )
            print("Created Collection, need to get data")

            text_chunks: list[str] = []
            for file in glob.glob("fauquier-zoning/*.pdf"):
                text = self.get_pdf_content(document=file)
                
                text_chunks += self.text_splitter.split_text(text)
                
            print(text_chunks)
            qdrant = QdrantVectorStore.from_texts(
                texts=text_chunks,
                embedding=self.text_embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key= os.getenv("QDRANT_API_KEY"),
                collection_name=ENV.COLLECTION_NAME,
            )
        else:
            qdrant = QdrantVectorStore.from_existing_collection(
                collection_name=ENV.COLLECTION_NAME,
                embedding=self.text_embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key= os.getenv("QDRANT_API_KEY"),
            )
        prompt_template = """
        You are an intelligent assistant tasked with answering user queries based on provided context. 
        Use the following context to respond to the user's question.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = (
            {"context": qdrant.as_retriever(), "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain    
    def get_pdf_content(self, document) -> str:
        raw_text: str = ""
        pdf_reader: PdfReader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

        return raw_text


    def get_chunks(self, text: str) -> list[str]:
        text_splitter: CharacterTextSplitter = CharacterTextSplitter(
            separator="\n", chunk_size=ENV.CHUNK_SIZE, chunk_overlap=200, length_function=len
        )
        text_chunks: list[str] = text_splitter.split_text(text)
        return text_chunks


def main():
    load_dotenv()
    chatbot = ChatBot()
    st.title("Interactive Chatbot built with Fauquier County Information")

    chain = chatbot.processing_local_data()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(chain.stream(input=prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
