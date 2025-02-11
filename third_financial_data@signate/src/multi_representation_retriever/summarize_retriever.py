## summarize_retrieverの部分を関数にまとめる
import uuid
from typing import Dict, List, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever


class SummarizeRetrieverManager:
    """企業ごとのretrieverを管理し、キャッシュを行うクラス"""

    def __init__(self):
        self.retrievers: Dict[str, MultiVectorRetriever] = {}
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstores: Dict[str, Chroma] = {}
        self.llm = ChatOpenAI(model="gpt-4o-mini", max_retries=0)

        # 要約用のchain
        self.summary_chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | self.llm
            | StrOutputParser()
        )

    def _split_text_into_chunks(self, filepath: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        return text_splitter.split_text(text)

    def _create_documents(self, chunks: List[str], title: str) -> List[Document]:
        """Create Document objects from text chunks"""
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": title,
                    "id": str(uuid.uuid4())
                }
            )
            for chunk in chunks
        ]

    def get_or_create_stores(self, md_path: str, company_name: str) -> Tuple[Optional[MultiVectorRetriever], Optional[Chroma]]:
        """
        企業名に対応するretrieverとvectorstoreを取得または作成
        """
        # 既存のretrieverとvectorstoreがあれば返す
        if company_name in self.retrievers and company_name in self.vectorstores:
            print(f"既存のretrieverとvectorstoreを使用します: {company_name}")
            return self.retrievers[company_name], self.vectorstores[company_name]

        print(f"新しいretrieverとvectorstoreを作成します: {company_name}")

        try:
            # vectorstoreの作成
            vectorstore = Chroma(
                collection_name="summarize",
                embedding_function=self.embedding_function
            )

            # retrieverの作成
            store = InMemoryByteStore()
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key="doc_id",
            )

            # テキストの処理
            title = md_path.split(".")[0].split("/")[-1]
            chunks = self._split_text_into_chunks(md_path)
            docs = self._create_documents(chunks, title)

            # 要約の生成
            summaries = self.summary_chain.batch(docs, {"max_concurrency": 5})

            # Document IDの生成
            doc_ids = [str(uuid.uuid4()) for _ in docs]

            # 要約とドキュメントのリンク
            summary_docs = [
                Document(page_content=s, metadata={"doc_id": doc_ids[i]})
                for i, s in enumerate(summaries)
            ]

            # retrieverとvectorstoreへの追加
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, docs)))

            # キャッシュに保存
            self.retrievers[company_name] = retriever
            self.vectorstores[company_name] = vectorstore

            return retriever, vectorstore

        except Exception as e:
            print(f"retrieverとvectorstoreの作成に失敗しました: {e}")
            return None, None

    def get_vectorstore(self, md_path: str, company_name: str) -> Optional[Chroma]:
        """企業名に対応するvectorstoreを取得"""
        _, vectorstore = self.get_or_create_stores(md_path, company_name)
        return vectorstore

    def get_retriever(self, md_path: str, company_name: str) -> Optional[MultiVectorRetriever]:
        """企業名に対応するretrieverを取得"""
        retriever, _ = self.get_or_create_stores(md_path, company_name)
        return retriever

    def contextual_search(self, query: str, md_path: str, company_name: str, k: int = 3) -> List[Document]:
        """コンテキストを考慮した検索を実行"""
        # クエリの拡張
        expand_prompt = ChatPromptTemplate.from_messages([
            ("system", "検索クエリを関連するキーワードで拡張してください。"),
            ("human", "{query}")
        ])

        expanded_query = expand_prompt | self.llm | StrOutputParser()
        expanded_query = expanded_query.invoke({"query": query})
        print(f"拡張されたクエリ: {expanded_query}")

        # 拡張クエリでの検索
        vectorstore = self.get_vectorstore(md_path, company_name)
        if not vectorstore:
            return []

        return vectorstore.similarity_search(expanded_query, k=k)

    def clear_cache(self, company_name: Optional[str] = None) -> None:
        """
        キャッシュをクリア

        Args:
            company_name (Optional[str]): クリアする企業名。Noneの場合は全てクリア
        """
        if company_name is None:
            # 全てのキャッシュをクリア
            self.retrievers.clear()
            print("全てのretrieverキャッシュをクリアしました")
        elif company_name in self.retrievers:
            # 指定された企業のキャッシュをクリア
            del self.retrievers[company_name]
            print(f"{company_name}のretrieverキャッシュをクリアしました")
        else:
            print(f"{company_name}のretrieverキャッシュは存在しません")


if __name__ == "__main__":
    from preprocess.pdf2markdown import PDFPreprocessor
    from multi_representation_retriever.summarize_retriever import SummarizeRetrieverManager

    # PDFファイルのディレクトリを指定
    pdf_dir = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/validation/documents"
    # ! SummarizeRetrieverManagerのインスタンスは最初に1回だけすること。
    # ! キャッシュで保持しているため、再度インスタンス化するとキャッシュがクリアされる。
    manager = SummarizeRetrieverManager()

    # プリプロセッサーを初期化
    preprocessor = PDFPreprocessor(pdf_dir)

    # クエリから会社を特定してPDFを変換
    query = "ダイドーグループのグループ会社の中で「株式会社」という文字を除き、ひらがなのみで構成された会社名"
    company_name, md_path, original_filename = preprocessor.find_company_and_convert(query)

    if company_name and md_path:
        print(f"処理が完了しました:")
        print(f"- 会社名: {company_name}")
        print(f"- Markdownファイル: {md_path}")
        print(f"- 元のPDFファイル: {original_filename}")


    # similarity_searchの使用
    vectorstore = manager.get_vectorstore(md_path, company_name)
    vector_docs = vectorstore.similarity_search(query, k=5)
    # contextual_searchの使用
    contextual_docs = manager.contextual_search(query=query, md_path=md_path, company_name=company_name, k=3)
    contextual_docs
