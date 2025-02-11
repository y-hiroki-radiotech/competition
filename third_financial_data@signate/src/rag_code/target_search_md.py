#!/usr/bin/env python
# coding: utf-8

# ### PDFをrenameしてLlama parserでIndex化

# In[1]:


from pdf_vectorize.create_llama_parse import LlamaParserVectorStore
from rename_pdf_file.rename_pdf import PdfRename


file_path = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/documents"


# In[ ]:


pdf_rename = PdfRename(file_path=file_path)
pdf_rename.rename_pdfs()


# In[3]:


import nest_asyncio

nest_asyncio.apply()

parser_openai = LlamaParserVectorStore(model="OpenAI", file_path=file_path, save_name="small")
# query_engine_openai = parser_openai.create_vector_engine(save=True)
query_engine_openai = parser_openai.load_vector_engine()


# In[4]:


## モデルをtext-embedding-3-largeに変更
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

new_embed_model = OpenAIEmbedding(model="text-embedding-3-large")
query_engine_openai._retriever._embed_model = new_embed_model
embed_model = query_engine_openai._retriever._embed_model
print(f"Current embedding model: {embed_model.__class__.__name__}")
print(f"Model configuration: {embed_model}")


# In[5]:


import pandas as pd

query_path = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/query.csv"
qa = pd.read_csv(query_path)


# ### 質問と回答を取り込む

# In[6]:


from typing import List

from langchain.schema.runnable import RunnableLambda
from langchain_core.documents import Document


def query_engine_with_docs(query: str) -> List[Document]:
    # LlamaIndexのレスポンスをLangChainのDocument型に変換
    response = query_engine_openai.query(query)

    # レスポンスをDocument型にラップ
    return [Document(
        page_content=response.response,
        metadata={"source": "query_engine"}
    )]

retriever_with_docs = RunnableLambda(query_engine_with_docs)


# ### direct_search_rag

# In[7]:


import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI


class CompanyFileFinder:
    def __init__(self, model="gpt-4o-mini", temperature=0.1, max_tokens=20):
        self.company_to_filepath = {}
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._setup_company_extractor()

    def _setup_company_extractor(self):
        template = """
        You are an entity extractor. Your role is to extract ONLY company names from questions.

        Instructions:
        - Extract ONLY single company name mentioned in the question
        - If no company name is present, respond with "分かりません"
        - If you are uncertain about the company name, respond with "分かりません"
        - If multiple company names are present (including merger/integration cases), respond with "分かりません"
        - Only provide the company name without any additional text

        Examples:
        Question: JR東日本の2023年度の営業収益は前年比でどのくらい増加しましたか？
        Answer: JR東日本

        Question: なぜ企業は環境に配慮した経営を重視するようになったのでしょうか？
        Answer: 分かりません

        Question: ソフトバンクグループの孫正義氏は何年に会長職に就任しましたか？
        Answer: ソフトバンクグループ

        Question: みずほ銀行と三井住友銀行の支店数を比較してください。
        Answer: 分かりません

        Question: イオンとセブン&アイの2022年度の売上高を比較してください。
        Answer: 分かりません

        Question: {input}
        Answer:"""

        self.company_extractor_prompt = ChatPromptTemplate.from_template(template)
        self.company_extractor_chain = (
            self.company_extractor_prompt
            | self.llm
            | StrOutputParser()
        )

    def extract_company_name(self, query: str) -> str:
        return self.company_extractor_chain.invoke(query)

    def extract_company_name_from_filename(self, filename: str) -> str:
        return filename.split('-')[0]

    def register_pdfs(self, pdf_directory: str) -> Dict[str, str]:
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                company_name = self.extract_company_name_from_filename(filename)
                filepath = os.path.join(pdf_directory, filename)
                self.company_to_filepath[company_name] = filepath
        return self.company_to_filepath

    def find_matching_filepath(self, query_name: str) -> Optional[str]:
        for company_name, filepath in self.company_to_filepath.items():
            if query_name in company_name or company_name in query_name:
                return filepath
        return None

    def get_pdf_path(self, query: str) -> Optional[str]:
        company_name = self.extract_company_name(query)

        if company_name == "分かりません":
            return None

        return self.find_matching_filepath(company_name)


# ### Markdownで保存したものに検索をかける

# In[8]:


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
    [
        ("system", template),
        ("human", "{input}"),
    ]
)

# Function to generate keywords
def generate_keywords(query: str) -> str:
    response = generate_keyword_prompt | llm | StrOutputParser()
    return response.invoke({"input": query})


# In[9]:


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
        return filename.split('-')[0]

    def get_md_path(self):
        """Get the corresponding markdown file path"""
        directory = os.path.dirname(self.pdf_path)
        base_name = os.path.basename(self.pdf_path)
        md_name = base_name.rsplit('.', 1)[0] + '.md'
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
            with open(md_path, 'r', encoding='utf-8') as f:
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
            with open(output_path, 'w', encoding='utf-8') as f:
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
                context_end = min(len(self.markdown_text.split()),
                                match_word_pos + self.context_words + 1)

                context = " ".join(self.markdown_text.split()[context_start:context_end])

                matches.append({
                    "keywords": keywords,
                    "context": context,
                    "position": start_pos,
                    "context_range": (context_start, context_end)
                })

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


# In[10]:


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

# Define the prompt template
prompt = ChatPromptTemplate.from_template(template)

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
                contexts.append(match['context'])

    # Process combination keyword results
    for keyword, matches in search_results.items():
        if " AND " in keyword:
            for match in matches:
                contexts.append(match['context'])

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
        response = chain.invoke({
            "input": query,
            "context": context
        })

        return response

    except Exception as e:
        print(f"Error processing query: {e}")
        return "処理中にエラーが発生しました。"


# In[24]:


from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def query_engine(query):
    response = query_engine_openai.query(query)
    return response.response


retriever = RunnableLambda(query_engine)

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


import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI


def summarize_to_max_tokens(text: str, model="gpt-4o-mini", temperature=0.1) -> str:
    """
    LLMを使用してテキストを54トークン以内に要約します。

    Args:
        text (str): 要約する元のテキスト
        model (str): 使用するLLMモデル
        temperature (float): LLMの温度パラメータ

    Returns:
        str: 54トークン以内に要約されたテキスト
    """
    # 既に54トークン以内の場合はそのまま返す
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= 54:
        return text

    # LLMの設定
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=54  # 出力を54トークンに制限
    )

    template = """
    あなたは簡潔な要約を作成するエキスパートです。
    与えられたテキストを、重要な情報を保持したまま54トークン以内で要約してください。

    要約のルール:
    - 必ず54トークン以内に収める
    - 重要な数値や固有名詞は保持する
    - 箇条書きの場合は「、」で区切って表現する
    - 不要な接続詞や副詞は削除する
    - 文末は「。」で終わる

    入力テキスト:
    {input_text}

    要約:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 要約を生成
    summary = chain.invoke({"input_text": text})

    # 要約が54トークンを超えている場合は切り詰める
    summary_tokens = enc.encode(summary)
    if len(summary_tokens) > 54:
        summary = enc.decode(summary_tokens[:54])
        # 最後の文字が途中で切れないように調整
        if not summary.endswith("。"):
            summary = summary.rsplit("、", 1)[0] + "。"

    return summary

# RAG チェーンの修正
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(summarize_to_max_tokens)  # 追加
)


# In[25]:


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


def summarize_to_max_tokens(text: str, model="gpt-4o-mini", temperature=0.1) -> str:
    """
    LLMを使用してテキストを54トークン以内に要約します。

    Args:
        text (str): 要約する元のテキスト
        model (str): 使用するLLMモデル
        temperature (float): LLMの温度パラメータ

    Returns:
        str: 54トークン以内に要約されたテキスト
    """
    # 既に54トークン以内の場合はそのまま返す
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= 54:
        return text

    # LLMの設定
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=54  # 出力を54トークンに制限
    )

    template = """
    あなたは簡潔な要約を作成するエキスパートです。
    与えられたテキストを、重要な情報を保持したまま54トークン以内で要約してください。

    要約のルール:
    - 必ず54トークン以内に収める
    - 重要な数値や固有名詞は保持する
    - 箇条書きの場合は「、」で区切って表現する
    - 不要な接続詞や副詞は削除する

    入力テキスト:
    {input_text}

    要約:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 要約を生成
    summary = chain.invoke({"input_text": text})

    # 要約が54トークンを超えている場合は切り詰める
    summary_tokens = enc.encode(summary)
    if len(summary_tokens) > 54:
        summary = enc.decode(summary_tokens[:54])
        # 最後の文字が途中で切れないように調整
        if not summary.endswith("。"):
            summary = summary.rsplit("、", 1)[0] + "。"

    return summary

# RAG チェーンの修正
contextual_rag_chain = (
    {"context": compression_retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(summarize_to_max_tokens))


# In[26]:


import weave
from tqdm import tqdm

weave.finish()
weave.init(project_name="第3回金融コンペ")


# In[27]:


# Usage example:
@weave.op()
def target_company_response(query):
    # 検索システムの初期化
    file_path = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/documents"
    finder = CompanyFileFinder()
    finder.register_pdfs(file_path)

    # 対象企業の特定
    filepath = finder.get_pdf_path(query)
    keywords = generate_keywords(query)

    if filepath and keywords:
        keywords = convert_string_to_list(keywords)

        print(f"Generated keywords: {keywords}")

        searcher = PDFSearcher(filepath)

        # Load content (will try MD first, then PDF)
        if searcher.load_content():
            # Search for keywords
            results = searcher.search_keyword_combination(keywords, window_size=100)
            response = process_query_with_context(query, results)

            if response is None:
                return contextual_rag_chain.invoke(query)

            if response in ["処理中にエラーが発生しました。", "分かりません"]:
                return rag_chain.invoke(query)

            return response

        return contextual_rag_chain.invoke(query)

    return contextual_rag_chain.invoke(query)


# In[28]:


from tqdm import tqdm


@weave.op()
def process():
    for i, query in tqdm(enumerate(qa["problem"]), total=len(qa["problem"]), position=0):
        query = qa["problem"][i]
        response = target_company_response(query)
        qa.at[i, "answer"] = response


# In[29]:


process()


# In[30]:


save_dir = '/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/submit/'
qa[["index", "answer"]].to_csv(save_dir + "predictions.csv", encoding="utf-8", index=False, header=False)


# In[ ]:




