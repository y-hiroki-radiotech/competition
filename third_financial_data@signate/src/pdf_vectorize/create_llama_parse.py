import asyncio
import os
from copy import deepcopy
from glob import glob
import logging
from typing import List, Tuple, Literal, Optional

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.schema import TextNode, Document
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_parse import LlamaParse
import sys

import settings

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

load_dotenv()

ModelType = Literal["OpenAI", "Gemini"]

class LlamaParserVectorStore:
    """
    PDFファイルを処理し、ベクトル検索エンジンを作成・管理するクラス。

    このクラスは、PDFファイルをパースし、OpenAIまたはGeminiモデルを使用して
    ベクトル検索エンジンを作成します。作成したエンジンの保存と読み込みも行えます。

    Attributes:
        llm: 言語モデル（OpenAIまたはGemini）
        embed_model: 埋め込みモデル
        file_paths (List[str]): 処理対象のPDFファイルパスのリスト
        reranker: Cohereの再ランキングプロセッサ
        save_and_load_path (str): インデックスの保存・読み込みパス
    """

    def __init__(self, model: ModelType = "OpenAI", file_path: Optional[str] = None, save_name: str = "") -> None:
        """
        初期化メソッド

        Args:
            model: 使用するモデルの種類（"OpenAI"または"Gemini"）
        """
        self.llm = self._initialize_llm(model)
        self.embed_model = self._initialize_embed(model)
        self.file_paths = self._file_paths(file_path)
        self.reranker = CohereRerank(top_n=5)
        self.save_and_load_path = f"./{model}_storage_{save_name}"

    def _initialize_llm(self, model: ModelType) -> OpenAI | Gemini:
        """
        言語モデルを初期化します。

        Args:
            model: 使用するモデルの種類

        Returns:
            初期化された言語モデル

        Raises:
            ValueError: サポートされていないモデルタイプが指定された場合
        """
        if model == "OpenAI":
            return OpenAI(model=settings.OPENAI_MODEL)
        elif model == "Gemini":
            return Gemini(model=settings.GEMINI_MODEL)
        else:
            raise ValueError("Unsupported model type. Use 'OpenAI' or 'Gemini'.")

    def _initialize_embed(self, model: ModelType) -> OpenAIEmbedding | GeminiEmbedding:
        """
        埋め込みモデルを初期化します。

        Args:
            model: 使用するモデルの種類

        Returns:
            初期化された埋め込みモデル

        Raises:
            ValueError: サポートされていないモデルタイプが指定された場合
        """
        if model == "OpenAI":
            return OpenAIEmbedding(model=settings.OPENAI_EMBEDDING_MODEL)
        elif model == "Gemini":
            return GeminiEmbedding(model=settings.GEMINI_EMBEDDING_MODEL)
        else:
            raise ValueError("Unsupported model type. Use 'OpenAI' or 'Gemini'.")

    def _file_paths(self, file_path) -> List[str]:
        """
        設定で指定されたディレクトリからPDFファイルのパスを取得します。

        Returns:
            PDFファイルパスのリスト

        Raises:
            FileNotFoundError: 指定されたパスにファイルが見つからない場合
        """
        if file_path is None:
            try:
                file_paths = glob(settings.pdf_file_path + "/*")
                print(file_paths)
                if not file_paths:
                    raise FileNotFoundError("No files found in the specified path.")
                return file_paths
            except Exception as e:
                print(e)
                return []
        else:
            try:
                file_paths = glob(file_path + "/*")
                print(file_paths)
                if not file_paths:
                    raise FileNotFoundError("No files found in the specified path.")
                return file_paths
            except Exception as e:
                print(e)
                return []


    def _get_page_nodes(self, docs: List[Document], separator: str = "\n---\n") -> List[TextNode]:
        """
        ドキュメントをページノードに分割します。

        Args:
            docs: 分割対象のドキュメントリスト
            separator: ページ区切りとして使用する文字列

        Returns:
            分割されたテキストノードのリスト
        """
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(text=doc_chunk, metadata=deepcopy(doc.metadata))
                nodes.append(node)
        return nodes

    def _save_index(self, index: VectorStoreIndex) -> None:
        """
        インデックスを指定されたパスに保存します。

        Args:
            index: 保存するベクトルストアインデックス
        """
        index.storage_context.persist(persist_dir=self.save_and_load_path)

    def _index_load(self) -> VectorStoreIndex:
        """
        保存されたインデックスを読み込みます。

        Returns:
            読み込まれたベクトルストアインデックス
        """
        storage_context = StorageContext.from_defaults(
            persist_dir=self.save_and_load_path
        )
        return load_index_from_storage(storage_context)

    def create_vector_engine(self, save: bool = False) -> BaseQueryEngine:
        """
        ベクトル検索エンジンを作成します。

        Args:
            save: 作成したインデックスを保存するかどうか

        Returns:
            作成されたクエリエンジン
        """
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        parser = LlamaParse(
            result_type="markdown",
            verbose=True,
            num_workers=4,
            parsing_instruction = "提供された文書はPDFである。1ページにはファイルのタイトル、2ページは目次となっており、それ以降のページは各ページごとの副題を持つ。資料は、テキスト、図、表、グラフ、イラストを含む。",
        )
        documents = parser.load_data(self.file_paths)

        node_parser = MarkdownElementNodeParser(
            llm=self.llm, num_worker=4
        )

        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        page_nodes = self._get_page_nodes(documents)

        recursive_index = VectorStoreIndex(nodes=base_nodes + objects + page_nodes)
        recursive_pages_query_engine = recursive_index.as_query_engine(
            similarity_top_k=5, node_postprocessor=[self.reranker], verbose=True
        )

        if save:
            self._save_index(recursive_index)

        return recursive_pages_query_engine

    def load_vector_engine(self) -> BaseQueryEngine:
        """
        保存されたベクトル検索エンジンを読み込みます。

        Returns:
            読み込まれたクエリエンジン

        Raises:
            FileNotFoundError: 保存されたインデックスが見つからない場合
        """
        if os.path.exists(self.save_and_load_path):
            index = self._index_load()
            return index.as_query_engine(
                similarity_top_k=5,
                node_postprocessor=[self.reranker], verbose=True
            )
        else:
            raise FileNotFoundError(
                f"No saved index found at {self.save_and_load_path}"
            )




if __name__ == "__main__":
    # OpenAIモデルを使用してベクトルエンジンを作成
    # ipynbで呼び出す場合
    import nest_asyncio
    nest_asyncio.apply()

    parser_openai = LlamaParserVectorStore(model="OpenAI")
    query_engine_openai = parser_openai.create_vector_engine(save=True)
    query = "何か質問をしてください"
    query_engine_openai.query(query)
    print("OpenAIモデルのベクトルエンジンが作成されました。")

    # # 保存されたOpenAIモデルのベクトルエンジンをロード
    # loaded_query_engine_openai = parser_openai.load_vector_engine()
    # print("保存されたOpenAIモデルのベクトルエンジンがロードされました。")

    # # Geminiモデルを使用してベクトルエンジンを作成
    # parser_gemini = LlamaParserVectorStore(model="Gemini")
    # query_engine_gemini = parser_gemini.create_vector_engine(save=True)
    # print("Geminiモデルのベクトルエンジンが作成されました。")

    # # 保存されたGeminiモデルのベクトルエンジンをロード
    # loaded_query_engine_gemini = parser_gemini.load_vector_engine()
    # print("保存されたGeminiモデルのベクトルエンジンがロードされました。")
