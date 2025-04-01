#!/usr/bin/env python
# coding: utf-8

# Install ollama at https://ollama.com/download then run command at terminal/cmd: ollama run deepseek-r1:1.5b

# In[1]:
get_ipython().system('pip3 install openai langchain faiss-cpu pandas jq langchain_openai')

import json
import requests
from langchain.llms.base import LLM
from pydantic import Field

class OllamaLLM(LLM):
    api_url: str = Field(default="http://localhost:11434/api/generate")

    @property
    def _llm_type(self) -> str:
        return "ollama_chat"
    
    def _call(self, prompt, stop=None) -> str:
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": prompt
        }
        response = requests.post(self.api_url, json=payload, stream=False)
        if response.status_code == 200:
            # Xử lý toàn bộ nội dung trả về thành text
            text_content = response.text.strip()
            result_text = ""
            # Tách từng dòng và giải mã JSON
            for line in text_content.splitlines():
                try:
                    data = json.loads(line)
                    result_text += data.get("response", "")
                except json.JSONDecodeError:
                    continue
            return result_text
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

    @property
    def _identifying_params(self) -> dict:
        return {"api_url": self.api_url}


# ## Load Data

# In[2]:


from langchain.document_loaders import JSONLoader

# Đường dẫn tới file JSON của bạn
file_path = "./ngu-linh-the-gioi-translated.json"
# Nếu file JSON của bạn có cấu trúc phức tạp, bạn có thể sử dụng jq_schema để trích xuất dữ liệu, ví dụ:
loader = JSONLoader(file_path=file_path, jq_schema=".[].content")
documents = loader.load()


# ## Chunk

# In[3]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Kích thước tối đa của mỗi đoạn
    chunk_overlap=100,    # Số ký tự trùng lặp giữa các đoạn
    separators=["\n\n", "</p>", "\n", " ", ""]  # Các dấu phân cách ưu tiên để tách văn bản
)

# Bước 3: Chia nhỏ nội dung văn bản
split_documents = text_splitter.split_documents(documents)


# ## Embedding & Vector store

# In[4]:


from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

ollama_llm = OllamaLLM(api_url="http://localhost:11434/api/generate")
# nếu không có HUGGINGFACE_TOKEN hoặc máy cấu hình thấp uncomment dòng này để sử dụng embedding all-MiniLM-L6-v2
# embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# nếu máy khoẻ và có HUGGINGFACE_TOKEN dùng Alibaba-NLP/gte-Qwen2-1.5B-instruct embedding vì context size lớn hơn, chất lượng cao và hỗ trợ tiếng việt
HUGGINGFACE_TOKEN = 'huggingface_token'
model_kwargs = {'trust_remote_code': True, 'token': HUGGINGFACE_TOKEN }
embedding_function = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", model_kwargs=model_kwargs)
vector_store = FAISS.from_documents(split_documents, embedding_function)
qa_chain = RetrievalQA.from_chain_type(llm=ollama_llm, retriever=vector_store.as_retriever())


# In[5]:


question = "Who is the main character in this story, and what are they like? Please provide a detailed description."
answer = qa_chain.invoke(question)
print(answer)


# In[ ]:




