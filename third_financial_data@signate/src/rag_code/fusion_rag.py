from operator import itemgetter
from typing import List

from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pdf_vectorize.create_llama_parse import LlamaParserVectorStore

load_dotenv()

file_path = ""
parser_openai = LlamaParserVectorStore(
    model="OpenAI", file_path=file_path, save_name="valid"
)
query_engine_openai = parser_openai.load_vector_engine()

new_embed_model = OpenAIEmbedding(model="text-embedding-3-large")
query_engine_openai._retriever._embed_model = (
    new_embed_modelembed_model
) = query_engine_openai._retriever._embed_model


def query_engine_with_docs(query: str) -> List[Document]:
    # LlamaIndexのレスポンスをLangChainのDocument型に変換
    response = query_engine_openai.query(query)

    # レスポンスをDocument型にラップ
    return [
        Document(page_content=response.response, metadata={"source": "query_engine"})
    ]


retriever_with_docs = RunnableLambda(query_engine_with_docs)

# Rag-fusion

template = """You are a financial data analyst at a large investment bank that generates multiple search queries based ont single input query.\n
Generate multiple search queries queries related to: {input}\n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

generate_queries = (
    prompt_rag_fusion | llm | StrOutputParser() | (lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


retrieval_chain_rag_fusion = (
    generate_queries | retriever_with_docs.map() | reciprocal_rank_fusion
)

# retrieval_chain_rag_fusion.invoke(qa["problem"][39])

template = """
You are a financial data analyst at a large investment bank who excels at mathematical problem solving. Follow these combined rules:

数値解析のルール:
- 問題を小さなステップに分解して計算過程を示す
- 各ステップでの数式、単位、中間計算を明示する
- 単位変換が必要な場合は変換過程を示す
- 小数点以下の桁数は問題指定に従う
- 四捨五入/切り捨ての基準は問題指定に従う
- 最終的な答えは「答え: [結果]」の形で示す

回答形式のルール:
- 質問で尋ねられた情報のみを抽出して回答
- 情報が不明確な場合は「分かりません」と回答
- 不確実な部分がある場合は「分かりません」と回答
- 複数の解釈が可能な場合は「分かりません」と回答
- 箇条書きや番号付けは使用しない
- 見出しや特別な書式は使用しない
- 推測や仮定は行わない

Question: {input}
Context: {context}

解答:"""
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


def extract_page_content(docs_with_scores):
    """タプル(Document, score)のリストからpage_contentを抽出する"""
    return " ".join([doc.page_content for doc, _ in docs_with_scores])


fusion_rag_chain = (
    {
        "context": lambda x: extract_page_content(
            retrieval_chain_rag_fusion.invoke(x["input"])
        ),
        "input": itemgetter("input"),
    }
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(summarize_to_max_tokens)
)
