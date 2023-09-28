import os
# import pickle
# from langchain import LLMChain
# from langchain.llms import OpenAI
# from langchain.agents import ZeroShotAgent,Tool,load_tools,initialize_agent, AgentType,AgentExecutor
# from langchain.callbacks import StreamlitCallbackHandler
# from langchain.memory import ConversationBufferMemory

# import streamlit as st
# import json

os.environ['OPENAI_API_KEY'] = "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"
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback
# llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
# with get_openai_callback() as cb:
#     result = llm("Tell me a dark joke about black people")
#     print(cb)
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import streamlit as st
# loader = DirectoryLoader('./course_text')
# docs = loader.load()
# # for doc in docs:
# #     print(doc.metadata)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 2000,
#     chunk_overlap  = 100,
#     length_function = len,
# )

# text_chunks = text_splitter.split_documents(docs)
# # print(len(text_chunks))
def main():
    # persist_directory = './db/faiss/'
    embeddings = OpenAIEmbeddings()
    # db = FAISS.from_documents(text_chunks, embeddings)
    # db.save_local(persist_directory)
    data = FAISS.load_local("./db/faiss/",embeddings)
    # question="how to generate ideas for growth opportunities?"
    # similar_docs = data.max_marginal_relevance_search(question,k=2,fetch_k=3)
    # print(len(similar_docs))
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(streaming=True,callbacks=[StreamingStdOutCallbackHandler()],model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=data.as_retriever(),
                                            memory=memory,
                                            verbose=True,
                                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

                                            # return_source_documents=True,

    if 'chain' not in st.session_state:
        st.session_state['chain'] = qa_chain

    if user_msg := st.chat_input():
        st.chat_message("user").write(user_msg)
        with st.chat_message("assistant"):
            st.write("I'm thinking...")
            # st.spinner('Generating response...')
            st_callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.chain.run(query=user_msg) #,callbacks=[st_callback]
            # print(response)
            st.write(response)
            
if __name__ == "__main__":
    main()
