import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema.runnable import RunnableLambda
from langchain.vectorstores import FAISS
from langchain_core.documents import Document


class CompanySearchSystem:
    def __init__(self, query_engine_openai):
        """
        検索システムを初期化します。

        Parameters:
            query_engine_openai: LlamaIndexのquery_engine
        """
        self.company_to_file_info = {}  # 会社名から[ファイルパス, 元のファイル名]へのマッピング
        self.ensemble_retriever = None
        self.documents = []
        self.query_engine = query_engine_openai
        self.embeddings = OpenAIEmbeddings()
        self.company_index = None

    def extract_company_name(self, filename: str) -> str:
        """PDFファイル名から会社名を抽出します"""
        return filename.split("-")[0]

    def initialize_company_index(self):
        """会社名のベクトルインデックスを作成します"""
        if not self.company_to_file_info:
            return

        company_names = list(self.company_to_file_info.keys())
        company_docs = [Document(page_content=name) for name in company_names]
        self.company_index = FAISS.from_documents(company_docs, self.embeddings)

    def register_pdfs(self, pdf_directory: str) -> Dict[str, Tuple[str, str]]:
        """
        指定されたディレクトリ内のPDFファイルを登録します

        Returns:
            Dict[str, Tuple[str, str]]: 会社名から(ファイルパス, 元のファイル名)へのマッピング
        """
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith(".pdf"):
                company_name = self.extract_company_name(filename)
                filepath = os.path.join(pdf_directory, filename)
                self.company_to_file_info[company_name] = (filepath, filename)
                print(f"登録: {company_name} (ファイル名: {filename})")

        # 会社名インデックスを初期化
        self.initialize_company_index()

        return self.company_to_file_info

    def get_original_filename(self, company_name: str) -> Optional[str]:
        """
        会社名から元のファイル名を取得します

        Parameters:
            company_name: 会社名

        Returns:
            Optional[str]: 元のファイル名。会社が見つからない場合はNone
        """
        if company_name in self.company_to_file_info:
            return self.company_to_file_info[company_name][1]
        return None

    def load_pdf_content(self, filepath: str) -> List[Document]:
        """PDFファイルの内容をDocument形式で読み込みます"""
        documents = []
        try:
            with pdfplumber.open(filepath) as pdf:
                content = "\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )

                paragraphs = content.split("\n\n")
                documents = [
                    Document(
                        page_content=para.strip(),
                        metadata={
                            "source": filepath,
                            "original_filename": os.path.basename(filepath),
                        },
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
            return [
                Document(
                    page_content=response.response, metadata={"source": "query_engine"}
                )
            ]

        return RunnableLambda(query_engine_with_docs)

    def _validate_weights(self, weights: List[float]) -> None:
        """重み付けの検証を行います"""
        if len(weights) != 2:
            raise ValueError("重み付けは2つの値である必要があります（LlamaIndexとBM25の重み）")

        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError("重み付けの合計は1である必要があります")

        if any(w < 0 for w in weights):
            raise ValueError("重み付けには負の値を使用できません")

    def create_retrievers(
        self, target_company: str, weights: List[float] = [0.5, 0.5]
    ) -> Optional[EnsembleRetriever]:
        """指定された会社のPDFからEnsembleRetrieverを作成します"""
        try:
            self._validate_weights(weights)
        except ValueError as e:
            print(f"重み付けが無効です: {str(e)}")
            return None

        if target_company not in self.company_to_file_info:
            print(f"エラー: {target_company} のPDFファイルが見つかりません。")
            return None

        filepath = self.company_to_file_info[target_company][0]
        self.documents = self.load_pdf_content(filepath)

        if not self.documents:
            print("文書の読み込みに失敗しました。")
            return None

        try:
            keyword_retriever = BM25Retriever.from_documents(self.documents, k=3)

            llama_retriever = self.create_llama_retriever()

            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[llama_retriever, keyword_retriever], weights=weights
            )

            print(
                f"Retrieverを作成しました（LlamaIndex重み: {weights[0]:.2f}, BM25重み: {weights[1]:.2f}）"
            )
            return self.ensemble_retriever

        except Exception as e:
            print(f"Retrieverの作成中にエラーが発生しました: {str(e)}")
            return None

    def find_target_company(
        self, query: str, similarity_threshold: float = 0.7
    ) -> Optional[str]:
        """
        クエリから対象企業をセマンティック検索で特定します

        Parameters:
            query: 検索クエリ
            similarity_threshold: 類似度の閾値（0.0 ~ 1.0）

        Returns:
            Optional[str]: 最も類似度の高い会社名。閾値を超える会社が見つからない場合はNone
        """
        if self.company_index is None:
            self.initialize_company_index()
            if self.company_index is None:
                return None

        # クエリに対して最も類似度の高い会社を検索
        results = self.company_index.similarity_search_with_score(query, k=1)

        if not results:
            return None

        company_doc, score = results[0]
        similarity = 1.0 - score  # FAISSのスコアを類似度に変換

        if similarity >= similarity_threshold:
            return company_doc.page_content

        return None


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define the prompt template for generating keywords
template = """
あなたは検索エキスパートとして、入力されたクエリから効果的な検索キーワードを生成します。

入力クエリ: {input}
会社名を除いたキーワードを3つ作成してください。
文字列のリストとして返してください。
例: ["A", "B", "C"]
"""

generate_keyword_prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", "{input}"),]
)

# Function to generate keywords
def generate_keywords(query: str) -> str:
    response = generate_keyword_prompt | llm | StrOutputParser()
    return response.invoke({"input": query})


import ast
import os
import re
from itertools import combinations

import pymupdf4llm


class PDFSearcher:
    def __init__(self, pdf_path, context_words=50):
        """
        Initialize PDFSearcher with a PDF file path and context window size
        Args:
            pdf_path (str): Path to the PDF file
            context_words (int): Number of context words for search results
        """
        self.pdf_path = pdf_path
        self.context_words = context_words
        self.markdown_text = None
        self.company_name = self._extract_company_name()

    def _extract_company_name(self):
        """Extract company name from the PDF filename"""
        filename = os.path.basename(self.pdf_path)
        return filename.split("-")[0]

    def get_md_path(self):
        """Get the corresponding markdown file path"""
        directory = os.path.dirname(self.pdf_path)
        base_name = os.path.basename(self.pdf_path)
        md_name = base_name.rsplit(".", 1)[0] + ".md"
        return os.path.join(directory, md_name)

    def load_content(self):
        """
        Load content from either MD or PDF file
        First tries to find and load an MD file, if not found, converts PDF to markdown
        Returns:
            bool: True if successful, False otherwise
        """
        md_path = self.get_md_path()

        # Try to load MD file first
        if os.path.exists(md_path):
            print(f"Found existing markdown file: {md_path}")
            return self._load_markdown(md_path)

        # If no MD file exists, convert PDF
        print(f"No markdown file found, converting PDF: {self.pdf_path}")
        return self.convert_to_markdown()

    def _load_markdown(self, md_path):
        """
        Load content from markdown file
        Args:
            md_path (str): Path to markdown file
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                self.markdown_text = f.read()
            print(f"Successfully loaded markdown from: {md_path}")
            return True
        except Exception as e:
            print(f"Error loading markdown file: {e}")
            return False

    def convert_to_markdown(self):
        """Convert PDF to markdown using pymupdf4llm"""
        try:
            self.markdown_text = pymupdf4llm.to_markdown(self.pdf_path)
            self.save_markdown()
            print(f"Successfully converted PDF to markdown")
            return True
        except Exception as e:
            print(f"Error converting PDF to Markdown: {e}")
            return False

    def save_markdown(self):
        """Save markdown content to file next to the PDF"""
        if self.markdown_text is None:
            print("No markdown text available. Run load_content first.")
            return

        output_path = self.get_md_path()
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.markdown_text)
            print(f"Markdown saved to: {output_path}")
        except Exception as e:
            print(f"Error saving markdown: {e}")

    def search_keyword_combination(self, keywords, window_size=None):
        """
        Search for combinations of keywords within a specified window
        Args:
            keywords (list): List of keywords to search for
            window_size (int, optional): Maximum words between keywords
        Returns:
            dict: Dictionary of search results organized by keyword combinations
        """
        if self.markdown_text is None:
            print("No markdown text available. Run load_content first.")
            return {}

        if window_size is None:
            window_size = self.context_words

        all_results = {}
        for r in range(1, len(keywords) + 1):
            for combo in combinations(keywords, r):
                combo_str = " AND ".join(combo)
                results = self._search_keyword_combination(combo, window_size)
                if results:
                    all_results[combo_str] = results

        return all_results

    def search_keyword_combination(self, keywords, window_size=None):
        """
        階層的に検索を行う - 3語、2語、1語の順で探索
        Args:
            keywords (list): List of keywords to search for
            window_size (int, optional): Maximum words between keywords
        Returns:
            dict: Dictionary of search results organized by keyword combinations
        """
        if self.markdown_text is None:
            print("No markdown text available. Run load_content first.")
            return {}

        if window_size is None:
            window_size = self.context_words

        all_results = {}

        # 最初に3語の組み合わせを試す
        if len(keywords) >= 3:
            for combo in combinations(keywords, 3):
                combo_str = " AND ".join(combo)
                results = self._search_keyword_combination(combo, window_size)
                if results:
                    all_results[combo_str] = results
                    return all_results  # 3語で結果が見つかったら終了

        # 3語で見つからなかった場合、2語の組み合わせを試す
        if not all_results and len(keywords) >= 2:
            for combo in combinations(keywords, 2):
                combo_str = " AND ".join(combo)
                results = self._search_keyword_combination(combo, window_size)
                if results:
                    all_results[combo_str] = results
                    return all_results  # 2語で結果が見つかったら終了

        # 2語でも見つからなかった場合、1語ずつ試す
        if not all_results:
            for keyword in keywords:
                combo = (keyword,)
                combo_str = keyword
                results = self._search_keyword_combination(combo, window_size)
                if results:
                    all_results[combo_str] = results

        return all_results

    def _search_keyword_combination(self, keywords, window_size):
        """
        特定のキーワードの組み合わせで検索
        Args:
            keywords (tuple): Tuple of keywords to search for
            window_size (int): Maximum words between keywords
        Returns:
            list: List of matches with context
        """
        text_lower = self.markdown_text.lower()
        matches = []

        first_keyword = keywords[0].lower()
        for match in re.finditer(first_keyword, text_lower):
            start_pos = match.start()

            words = self.markdown_text[start_pos:].split()
            window_text = " ".join(words[:window_size]).lower()

            all_found = True
            for keyword in keywords[1:]:
                if keyword.lower() not in window_text:
                    all_found = False
                    break

            if all_found:
                match_word_pos = len(self.markdown_text[:start_pos].split())
                context_start = max(0, match_word_pos - self.context_words)
                context_end = min(
                    len(self.markdown_text.split()),
                    match_word_pos + self.context_words + 1,
                )

                context = " ".join(
                    self.markdown_text.split()[context_start:context_end]
                )

                matches.append(
                    {
                        "keywords": keywords,
                        "context": context,
                        "position": start_pos,
                        "context_range": (context_start, context_end),
                    }
                )

        return matches


def print_combination_results(results):
    """
    Pretty print combination search results
    Args:
        results (dict): Search results organized by keyword combinations
    """
    if not results:
        print("No matches found.")
        return

    print("\nSearch Results:")
    print("=" * 80)

    for combo, matches in results.items():
        print(f"\nKeyword Combination: {combo}")
        print("-" * 80)
        print(f"Found {len(matches)} matches:")

        for i, match in enumerate(matches, 1):
            print(f"\nMatch {i}:")
            print(f"Keywords: {', '.join(match['keywords'])}")
            print(f"Context: ...{match['context']}...")

        print("-" * 80)


# 文字列からリストへの変換
def convert_string_to_list(keywords_string):
    """
    '["2030", "ありたい姿", "ビジョン"]' のような文字列を
    実際のリストに変換します
    """
    try:
        # ast.literal_eval を使用して安全に文字列をリストに変換
        keywords = ast.literal_eval(keywords_string)
        return keywords
    except Exception as e:
        print(f"Error converting string to list: {e}")
        return []


import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial data analyst at a large investment bank. Your role is to provide ONLY the key information that directly answers the question, without any additional context or analysis.

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
- Do not speculate or make assumptions about missing information""",
        ),
        ("human", "Question: {input}\nContext: {context}"),
    ]
)

# Set up the chain
chain = prompt | llm | StrOutputParser() | summarize_to_max_tokens


def format_search_context(search_results):
    """
    Format search results into a single context string
    Args:
        search_results (dict): Dictionary containing search results
    Returns:
        str: Formatted context string
    """
    contexts = []

    # Process single keyword results
    for keyword, matches in search_results.items():
        # Skip combination results
        if " AND " not in keyword:
            for match in matches:
                contexts.append(match["context"])

    # Process combination keyword results
    for keyword, matches in search_results.items():
        if " AND " in keyword:
            for match in matches:
                contexts.append(match["context"])

    return "\n---\n".join(contexts)


def process_query_with_context(query: str, context_dict: dict):
    """
    Process a query using the provided context
    Args:
        query (str): Query string
        context_dict (dict): Dictionary containing search results
    Returns:
        str: Response from the chain
    """
    try:
        # Format the context
        context = format_search_context(context_dict)

        # Process with LangChain
        response = chain.invoke({"input": query, "context": context})

        return response

    except Exception as e:
        print(f"Error processing query: {e}")
        return "処理中にエラーが発生しました。"


# Usage example:
def target_company_response(query):
    # 検索システムの初期化
    file_path = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/validation/documents"
    search_system = CompanySearchSystem(query_engine_openai)

    # PDFファイルの登録
    search_system.register_pdfs(file_path)

    # 対象企業の特定
    target_company = search_system.find_target_company(query)

    if target_company:
        print(f"\n対象企業: {target_company}")

        # Retrieverの作成と検索の実行
        ensemble_retriever = search_system.create_retrievers(
            target_company, weights=[0.5, 0.5]
        )
        if ensemble_retriever:
            ensemble_result = ensemble_retriever.get_relevant_documents(query)
            filename = ensemble_result[1].metadata.get("original_filename", "不明")

    keywords = generate_keywords(query)
    keywords = convert_string_to_list(keywords)
    print(f"Generated keywords: {keywords}")

    pdf_path = f"{file_path}/{filename}"
    searcher = PDFSearcher(pdf_path)

    # Load content (will try MD first, then PDF)
    if searcher.load_content():
        # Search for keywords
        results = searcher.search_keyword_combination(keywords, window_size=100)
        print_combination_results(results)

    response = process_query_with_context(query, results)
    return response


if __name__ == "__main__":
    from tqdm import tqdm
    import weave

    weave.init(project_name="target_search_md_RAG")
    @weave.op()
    def process():
        for i, query in tqdm(enumerate(qa["problem"]), total=len(qa["problem"]), position=0):
            query = qa["problem"][i]
            response = target_company_response(query)
            qa.at[i, "target_company_rag"] = response

    process()

