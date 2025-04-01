#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install openai langchain faiss-cpu pandas jq langchain_openai')


# ## Import data

# In[2]:


from langchain.document_loaders import JSONLoader

# Đường dẫn tới file JSON của bạn
file_path = "./ngu-linh-the-gioi-translated.json"

# Nếu file JSON của bạn có cấu trúc phức tạp, bạn có thể sử dụng jq_schema để trích xuất dữ liệu, ví dụ:
# loader = JSONLoader(file_path=file_path, jq_schema=".data")

loader = JSONLoader(file_path=file_path, jq_schema=".[].content")
documents = loader.load()

# for doc in documents[0:5]:
#     print(doc.page_content)


# ## Chunk documents

# In[3]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,      # Kích thước tối đa của mỗi đoạn
    chunk_overlap=300,    # Số ký tự trùng lặp giữa các đoạn
    separators=["\n\n", "</p>", "\n", " ", ""]  # Các dấu phân cách ưu tiên để tách văn bản
)

# Bước 3: Chia nhỏ nội dung văn bản
split_documents = text_splitter.split_documents(documents)

# In ra các đoạn văn bản sau khi chia nhỏ
# for doc in split_documents[0:5]:
#     print('----------------')
#     print(doc.page_content)


# ## Embedding

# In[4]:


from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

OPEN_AI_API_KEY='api_key'

embeddings = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY)
vector_store = FAISS.from_documents(split_documents, embeddings)


# ## Build Chain

# In[5]:


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4o",  # Chỉ định sử dụng GPT-4o
    temperature=0, # temperature từ 0-1 càng lớn thì càng sáng tạo hơn
    openai_api_key=OPEN_AI_API_KEY  # Thay bằng API key của bạn
)
# llm = OpenAI(api_key=OPEN_AI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())


# ## Handle Input

# In[6]:


user_question = "Nhân vật chính trong câu chuyện này là ai và là người như thế nào, hãy mô tả chi tiết"
# vector_store.
response = qa_chain.invoke(user_question)
print(response)


# In[ ]:




