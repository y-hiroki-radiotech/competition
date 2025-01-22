from dotenv import load_dotenv
import settings
from glob import glob
from pdf_reader import pdf_reader_run
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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