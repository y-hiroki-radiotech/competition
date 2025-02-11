from operator import itemgetter
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()


def query_engine_with_docs_multi(query: str) -> List[Document]:
    if isinstance(query, dict):
        query = query.get("input", "")

    # responseがNoneの場合のハンドリングを追加
    response = query_engine_openai.query(query)
    if response is None or response.response is None:
        return [Document(page_content="No response found", metadata={"source": "query_engine"})]

    return [Document(
        page_content=response.response,
        metadata={"source": "query_engine"}
    )]

retriever_with_docs_multi = RunnableLambda(query_engine_with_docs_multi)


template = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {input}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_perspectives
    | ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def get_queries_and_docs(query: str):
    try:
        if isinstance(query, dict):
            query = query.get("input", "")

        generated_queries = generate_queries.invoke({"input": query})
        if generated_queries is None:
            generated_queries = []

        all_queries = [query] + generated_queries

        combined_qa = []
        all_docs = []

        for q in all_queries:
            try:
                docs = retriever_with_docs_multi.invoke(q)
                if docs:
                    combined_answer = " ".join([
                        doc.page_content for doc in docs if doc and doc.page_content
                    ])
                    combined_qa.append(f"質問：{q}\n回答：{combined_answer}")
                    all_docs.extend(docs)
            except Exception as e:
                print(f"Error processing query {q}: {e}")
                continue

        unique_docs = get_unique_union([all_docs]) if all_docs else []

        return {
            "context": unique_docs,
            "combined_qa": combined_qa,
            "input": query
        }
    except Exception as e:
        print(f"Error in get_queries_and_docs: {e}")
        return {
            "context": [],
            "combined_qa": [],
            "input": query
        }


multi_template = """
あなたは専門知識を持つデータアナリストです。以下の質問に対し、関連する文書を基に **直接的で簡潔な回答** を提供してください。

## **回答方針**
1. **質問の意図に合致する情報のみを抜き出す。**
2. **説明や補足を省略し、簡潔なフレーズまたは数値のみで回答する。**
3. **単位（％、拠点など）は必要に応じて明示する。**
4. **リスト形式が適切な場合は「, 」区切りの文字列で出力する。**
5. **取得した文書内に明確な情報がない場合は「分からない」と回答する。**

---

## **質問**
{input}

## **過去の質問と回答**
{combined_qa}

## **取得された文書**
{context}

---

**Answer:**
（ここに最適な回答を出力）
"""

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
multi_template = ChatPromptTemplate.from_template(multi_template)

final_rag_chain = (
    RunnablePassthrough.assign(queries_and_docs=get_queries_and_docs)
    | {
        "input": itemgetter("input"),
        "context": lambda x: "\n".join(
            doc.page_content for doc in x["queries_and_docs"]["context"]
        ),
        "combined_qa": lambda x: "\n\n".join(x["queries_and_docs"]["combined_qa"]),
    }
    | multi_template
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    question = qa["problem"][0]
    print(question)
    response = final_rag_chain.invoke({"input": question})
    response
    from tqdm import tqdm

    col = "multi-query-rag"

    @weave.op()
    def process():
        for i, question in tqdm(enumerate(qa["problem"]), total=len(qa["problem"]), position=0):
            count = 1
            while True:
                try:
                    print("==multi-query-rag==")
                    response = final_rag_chain.invoke({"input": question})
                    qa.at[i, col] = response
                    break  # 成功したらwhileループを抜ける

                except Exception as e:
                    print(f"Error processing input {i}: {e}")
                    print("Retrying...")
                    count += 1
                    if count > 3:
                        response = contextual_rag_chain.invoke({"input": question})
                        qa.at[i, col] = response
                        break
                    continue  # エラーが発生したら再試行
