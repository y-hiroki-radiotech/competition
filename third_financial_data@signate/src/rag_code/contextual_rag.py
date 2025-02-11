from typing import List

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv()

def query_engine_with_docs(query: str) -> List[Document]:
    # LlamaIndexのレスポンスをLangChainのDocument型に変換
    response = query_engine_openai.query(query)

    # レスポンスをDocument型にラップ
    return [Document(
        page_content=response.response,
        metadata={"source": "query_engine"}
    )]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

retriever_with_docs = RunnableLambda(query_engine_with_docs)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever_with_docs
)


template = """
You are a financial data analyst at a large investment bank. Your role is to provide ONLY the key information that directly answers the question, without any additional context or analysis.

Instructions:
- Extract ONLY the specific information asked in the question
- If the information cannot be found or is unclear in the given context, respond with "分かりません"
- If you are uncertain about any part of the answer, respond with "分かりません"
- If multiple interpretations are possible, respond with "分かりません"
- For clear answers, provide them in Japanese
- List items with commas (、) as separators
- Do not include any explanations, descriptions, or analysis
- Do not use bullet points or numbering
- Do not add any headers or formatting
- Do not speculate or make assumptions about missing information

Question: {input}
Context: {context}
Provide only the direct answer:"""

prompt = ChatPromptTemplate.from_template(template)


def summarize_to_max_tokens(text: str) -> str:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    # 既に54トークン以下の場合はそのまま返す
    if len(tokens) <= 54:
        return text

    # 回答が長い場合、カンマ区切りの項目を優先して短縮
    items = text.split("、")
    current_text = ""

    for item in items:
        temp_text = current_text + ("、" if current_text else "") + item
        temp_tokens = enc.encode(temp_text)
        if len(temp_tokens) <= 54:
            current_text = temp_text
        else:
            break

    return current_text

# RAG チェーンの修正
contextual_rag_chain = (
    {"context": compression_retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(summarize_to_max_tokens))
