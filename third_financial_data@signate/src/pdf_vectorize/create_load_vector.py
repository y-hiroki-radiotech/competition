from glob import glob
import os
import settings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pdf_vectorize.pdf_reader import pdf_reader_run


def create_vector():
    """
    ドキュメントのリストからベクトルストアのレトリーバーを作成します。
    この関数は、再帰的な文字テキストスプリッターを使用してドキュメントをチャンクに分割し、
    Google Generative AI 埋め込みを使用してチャンクの埋め込みを生成し、
    それらの埋め込みを FAISS ベクトルストアに保存します。
    その後、ベクトルストアのレトリーバーオブジェクトを返します。
    戻り値:
        retriever: FAISS ベクトルストアのレトリーバーオブジェクト。
    """
    load_dotenv()
    try:
        file_paths = glob(settings.pdf_file_path+"/*")
    except Exception as e:
        print(e)
    docs_list = pdf_reader_run(file_paths)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs_list)

    gen_embedding = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
    gen_faiss_vectorstore = FAISS.from_documents(
                        documents=splits,
                        embedding=gen_embedding
                    )
    retriever = gen_faiss_vectorstore.as_retriever()

    return retriever

def load_vector(file_path: str="./gen_faiss_index"):
    loaded_vectorstore = FAISS.load_local(
    file_path,
    embeddings=GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL),
    allow_dangerous_deserialization=True,
    )

    # retrieverとして使用
    loaded_retriever = loaded_vectorstore.as_retriever()

    return loaded_retriever