from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain.schema.runnable import RunnableLambda
import pdfplumber
import os
from typing import List, Optional

class CompanySearchSystem:
    def __init__(self, query_engine_openai):
        """
        検索システムを初期化します。LlamaIndexのquery_engineとBM25Retrieverを組み合わせて
        より包括的な検索を実現します。

        Parameters:
            query_engine_openai: LlamaIndexのquery_engine
        """
        self.company_to_file = {}
        self.ensemble_retriever = None
        self.documents = []
        self.query_engine = query_engine_openai

    def extract_company_name(self, filename: str) -> str:
        """PDFファイル名から会社名を抽出します"""
        return filename.split('-')[0]

    def register_pdfs(self, pdf_directory: str) -> None:
        """指定されたディレクトリ内のPDFファイルを登録します"""
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                company_name = self.extract_company_name(filename)
                filepath = os.path.join(pdf_directory, filename)
                self.company_to_file[company_name] = filepath
                # print(f"登録: {company_name}")

    def load_pdf_content(self, filepath: str) -> List[Document]:
        """PDFファイルの内容をDocument形式で読み込みます"""
        documents = []
        try:
            with pdfplumber.open(filepath) as pdf:
                content = '\n'.join(
                    page.extract_text()
                    for page in pdf.pages
                    if page.extract_text()
                )

                paragraphs = content.split('\n\n')
                documents = [
                    Document(
                        page_content=para.strip(),
                        metadata={"source": filepath}
                    )
                    for para in paragraphs
                    if para.strip()
                ]
        except Exception as e:
            print(f"PDFの読み込み中にエラーが発生しました: {str(e)}")
        return documents

    def create_llama_retriever(self) -> RunnableLambda:
        """LlamaIndexのquery_engineをRetriever形式にラップします"""
        def query_engine_with_docs(query: str) -> List[Document]:
            response = self.query_engine.query(query)
            return [Document(
                page_content=response.response,
                metadata={"source": "query_engine"}
            )]

        return RunnableLambda(query_engine_with_docs)

    def _validate_weights(self, weights: List[float]) -> None:
        """
        検索の重み付けが有効かどうかを検証します。

        Parameters:
            weights: 検証する重み付けのリスト。LlamaIndexとBM25の重みを表します。

        Raises:
            ValueError: 重み付けが無効な場合（合計が1でない、負の値を含む、または要素数が不正な場合）
        """
        if len(weights) != 2:
            raise ValueError("重み付けは2つの値である必要があります（LlamaIndexとBM25の重み）")

        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError("重み付けの合計は1である必要があります")

        if any(w < 0 for w in weights):
            raise ValueError("重み付けには負の値を使用できません")

    def create_retrievers(self, target_company: str, weights: List[float]=[0.5, 0.5]) -> Optional[EnsembleRetriever]:
        """
        指定された会社のPDFからEnsembleRetrieverを作成します。
        LlamaIndexのRetrieverとBM25Retrieverを指定された重みで組み合わせます。

        Parameters:
            target_company: 検索対象の会社名
            weights: [LlamaIndex重み, BM25重み]の形式で指定する重み付け

        Returns:
            Optional[EnsembleRetriever]: 作成されたRetriever。エラー時はNone
        """
        # 重み付けの検証
        try:
            self._validate_weights(weights)
        except ValueError as e:
            print(f"重み付けが無効です: {str(e)}")
            return None

        # 対象会社のPDFファイルの存在確認
        if target_company not in self.company_to_file:
            print(f"エラー: {target_company} のPDFファイルが見つかりません。")
            return None

        # PDFファイルからテキストを抽出
        filepath = self.company_to_file[target_company]
        self.documents = self.load_pdf_content(filepath)

        if not self.documents:
            print("文書の読み込みに失敗しました。")
            return None

        try:
            # キーワードベースの検索機能（BM25）を作成
            keyword_retriever = BM25Retriever.from_documents(
                self.documents,
                k=3  # 上位3件の結果を取得
            )

            # 意味ベースの検索機能（LlamaIndex）を作成
            llama_retriever = self.create_llama_retriever()

            # 両方の検索機能を組み合わせて最終的なRetrieverを作成
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[llama_retriever, keyword_retriever],
                weights=weights  # 指定された重みを使用
            )

            print(f"Retrieverを作成しました（LlamaIndex重み: {weights[0]:.2f}, BM25重み: {weights[1]:.2f}）")
            return self.ensemble_retriever

        except Exception as e:
            print(f"Retrieverの作成中にエラーが発生しました: {str(e)}")
            return None

    def find_target_company(self, query: str) -> Optional[str]:
        """クエリから対象企業を特定します"""
        target_company = None
        max_length = 0

        for company in self.company_to_file.keys():
            if company in query and len(company) > max_length:
                target_company = company
                max_length = len(company)

        return target_company

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import tiktoken

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def summarize_to_max_tokens(text: str) -> str:
    """Summarize text to maximum token length"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= 54:
        return text

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

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial data analyst at a large investment bank. Your role is to provide ONLY the key information that directly answers the question, without any additional context or analysis.

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
- Do not speculate or make assumptions about missing information"""),
    ("human", "Question: {input}\nContext: {context}")
])

# Set up the chain
chain = prompt | llm | StrOutputParser() | summarize_to_max_tokens


def process_query_with_context(query: str, results: dict):
    """
    Process a query using the provided context
    Args:
        query (str): Query string
        results (List): List containing search results
    Returns:
        str: Response from the chain
    """
    try:
        # Format the context
        context = results[0].page_content

        # Process with LangChain
        response = chain.invoke({
            "input": query,
            "context": context
        })

        return response

    except Exception as e:
        print(f"Error processing query: {e}")
        return "処理中にエラーが発生しました。"

def bm25_rag_response(query):
    # 検索システムの初期化
    file_path = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/validation/documents"
    search_system = CompanySearchSystem(query_engine_openai)

    # PDFファイルの登録
    search_system.register_pdfs(file_path)

    # 対象企業の特定と検索の実行
    target_company = search_system.find_target_company(query)
    if target_company:
        print(f"\n対象企業: {target_company}")

        ensemble_retriever = search_system.create_retrievers(target_company, weights=[0.5, 0.5])
        if ensemble_retriever:
            results = ensemble_retriever.get_relevant_documents(query)

    response = process_query_with_context(query, results)

    return response



if __name__ == "__main__":
    def process():
    for i, query in tqdm(enumerate(qa["problem"]), total=len(qa["problem"]), position=0):
        response = bm25_rag_response(query)
        qa.at[i, "ans_fusion_rag"] = response

    process()
