import operator
from typing import Annotated, List, Any, Dict

from pydantic import BaseModel, Field

class State(BaseModel):
    query: str = Field(..., description="ユーザからの質問")
    current_role: str = Field(default="", description="選定された回答ロール")
    messages_summarized: Annotated[List[str], operator.add] = Field(default=[], description="multi-representationの回答履歴")
    messages_retrieved: Annotated[List[str], operator.add] = Field(default=[], description="検索の回答履歴")
    retrieved_documents: List[str] = Field(default=[], description="検索によって取得されたドキュメント")
    summarized_documents: List[str] = Field(default=[], description="multi-representationによって取得されたドキュメント")
    current_judge: bool = Field(default=False, description="回答の品質の結果")
    judgement_reason: str = Field(default="", description="品質チェックの判定")

from preprocess.pdf2markdown import PDFPreprocessor
from multi_representation_retriever.summarize_retriever import SummarizeRetrieverManager

manager = SummarizeRetrieverManager()

def vector_search_retrieval(state: State) -> dict[str, Any]:
    query = state.query
    pdf_dir = state.pdf_dir

    # プリプロセッサーを初期化
    preprocessor = PDFPreprocessor(pdf_dir)

    # クエリから会社を特定してPDFを変換
    company_name, md_path, original_filename = preprocessor.find_company_and_convert(query)

    if company_name and md_path:
        print(f"処理が完了しました:")
        print(f"- 会社名: {company_name}")
        print(f"- Markdownファイル: {md_path}")
        print(f"- 元のPDFファイル: {original_filename}")
    try:
        contextual_docs = manager.contextual_search(query=query, md_path=md_path, company_name=company_name, k=3)
    except Exception as e:
        print(f"Error: {e}")
        contextual_docs = None
    return {"retrieved_documents": contextual_docs}


ROLES = {
  "1": {
    "name": "財務分析エキスパート",
    "description": "企業の財務情報や経営指標に関する質問に答える",
    "details": "決算情報、経営指標、財務比率などの分析と解釈を行い、正確な数値情報と共に意味のある洞察を提供します。"
  },
  "2": {
    "name": "企業情報スペシャリスト",
    "description": "企業の組織体制、人事、事業構造に関する質問に答える",
    "details": "企業の組織構造、人事情報、事業部門の構成など、企業の基本情報に関する詳細な知識を提供します。"
  },
  "3": {
    "name": "市場動向アナリスト",
    "description": "業界動向や市場データに関する質問に答える",
    "details": "業界統計、市場シェア、消費動向など、マクロ的な市場情報の分析と解釈を提供します。"
  },
  "4": {
    "name": "事業戦略コンサルタント",
    "description": "企業の戦略や施策に関する質問に答える",
    "details": "企業の経営戦略、事業計画、重点施策などについて、背景や意図を含めた分析を提供します。"
  },
  "5": {
    "name": "データ比較アナリスト",
    "description": "複数の指標や期間の比較分析に関する質問に答える",
    "details": "時系列データの比較、複数指標間の関係性分析、比率計算など、数値データの比較分析を行います。数値の計算が必要な場合はステップバイステップで考え、定義を確認すること。"
  }
}

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))

def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join([f"{k}. {v['name']: {v['description']}}" for k, v in ROLES.items()])
    # role_options = "\n".join([f"{k}. {v['name']} - {v['description']}" for k, v in ROLES.items()])


    prompt = ChatPromptTemplate.from_template(
"""あなたは優秀な金融アナリストです。質問を分析し、最も適切な回答担当ロールを選択してください。

選択肢:
{role_options}

回答は選択肢の番号（1-5）のみを返してください。

質問: {query}
""".strip()
    )
    # 選択肢の番号のみを返すことを期待したいため、max_tokensの値を1に変更
    chain = prompt | llm.with_config(Configurable=dict(max_toknes=1)) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})

    selected_role = ROLES[role_number.strip()]["name"]
    return {"current_role": selected_role}


def answering_node_summarized(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    retrieved_documents = state.retrieved_documents
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])

    prompt = ChatPromptTemplate.from_template(
"""あなたは{role}として回答してください。以下の質問に対して、回答方針を満たすように回答してください。検索結果に基づいて回答をお願いします。
## **回答方針**
1. **質問の意図に合致する情報のみを抜き出す。**
2. **説明や補足を省略し、簡潔な単語または数値のみで回答する。**
3. **単位（％、拠点など）は必要に応じて明示する。**
4. **リスト形式が適切な場合は「, 」区切りの文字列で出力する。**
5. **取得した文書内に明確な情報がない場合は「分から

役割の詳細:
{role_details}

質問: {query}
検索結果: {context}

回答:""".strip()
    )
    # 検索結果をプロンプトのコンテキストに組み込む (複数ドキュメントを結合)
    context_str = "\n\n".join(retrieved_documents)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role, "role_details": role_details, "query": query, "context": context_str})
    return {"messages_summarized": [answer]}


def answering_node_retrieval(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    summarized_documents = state.summarized_documents
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])

    prompt = ChatPromptTemplate.from_template(
"""あなたは{role}として回答してください。以下の質問に対して、回答方針を満たすように回答してください。検索結果に基づいて回答をお願いします。
## **回答方針**
1. **質問の意図に合致する情報のみを抜き出す。**
2. **説明や補足を省略し、簡潔な単語または数値のみで回答する。**
3. **単位（％、拠点など）は必要に応じて明示する。**
4. **リスト形式が適切な場合は「, 」区切りの文字列で出力する。**
5. **取得した文書内に明確な情報がない場合は「分から

役割の詳細:
{role_details}

質問: {query}
検索結果: {context}

回答:""".strip()
    )
    # 検索結果をプロンプトのコンテキストに組み込む (複数ドキュメントを結合)
    context_str = "\n\n".join(summarized_documents)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role, "role_details": role_details, "query": query, "context": context_str})
    return {"messages_retrieved": [answer]}

class Judgement(BaseModel):
    judge: bool = Field(default=False, description="判定結果")
    reason: str = Field(default="", description="判定理由")

def check_node(state: State) -> Dict[str, Any]:
    query = state.query
    answer = state.messages[-1]
    prompt = ChatPromptTemplate.from_template(
"""以下の回答が、質問に対して適切な基準を満たしているかを判定してください。
基準を満たしている場合は 'True'、満たしていない場合は 'False' を返してください。
また、満たしていない場合は、その理由を具体的に説明してください。

**評価基準**
1. 回答の十分性:
   - 数値やカテゴリーの単純な回答の場合、それ自体が完全な回答となり得ます
   - 質問の意図に沿った情報が含まれていれば十分とみなします

2. 形式の適切性:
   - 簡潔な回答は、それが質問に答えていれば適切とみなします
   - 回答が53トークン以下であることを確認します


### 評価対象
質問: {query}
# 質問の意図をじっくり考えましょう。それを踏まえて回答を評価してください。
回答: {answer}

### **出力フォーマット**""".strip()
    )
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }

from langchain_core.documents import Document
from langchain.schema.runnable import RunnableLambda
from typing import List


def retrieve(state: State) -> List[str]:
    query = state.query

    if isinstance(query, dict):
        query = query.get("input", "")

    # responseがNoneの場合のハンドリングを追加
    response = query_engine_openai.query(query)

    if response is None or response.response is None:
        retrieved_docs = ["No response found"]
    else:
        retrieved_docs = [response.response]  # ← Document ではなく `str` のリストにする

    return {"retrieved_documents": retrieved_docs}

from langgraph.graph import StateGraph

workflow = StateGraph(State)
# RAG関連のノードを追加
workflow.add_node("selection", selection_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("vector_search_retrieval", vector_search_retrieval)
workflow.add_node("answering_retrieval", answering_node_retrieval)
workflow.add_node("answering_summarized", answering_node_summarized)
workflow.add_node("check", check_node)

from langgraph.graph import END
# selectionノードから処理を開始
workflow.set_entry_point("selection")

# selectionノードからansweringノードへのエッジを追加
workflow.add_edge("selection", "answering_retrieval")
workflow.add_edge("selection", "answering_summarized")
# selectionノードからの分岐を追加
workflow.add_edge("selection", "retrieve")  # この行を追加
workflow.add_edge("selection", "vector_search_retrieval")
# retrieveノードからansweringノードへのエッジを追加
workflow.add_edge("retrieve", "answering_retrieval")
workflow.add_edge("vector_search_retrieval", "answering_summarized")
# answeringノードからcheckノードへのエッジを追加
workflow.add_edge("answering_retrieval", "check")
workflow.add_edge("answering_summarized", "check")
# checkノードから次のノードへの遷移に条件付きエッジを定義
# state.current_judgeの値がTrueの場合はENDノードへ遷移, Falseの場合はretrieveノードへ遷移
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {True: END, False: "selection"}
)
