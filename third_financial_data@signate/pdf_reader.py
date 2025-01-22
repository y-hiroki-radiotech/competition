from langchain_community.document_loaders import (
    PDFPlumberLoader, 
    UnstructuredPDFLoader, 
    PDFMinerLoader,
    PyPDFLoader
)
from typing import List
from langchain_community.document_loaders import Document
from glob import glob

def load_pdf_with_multiple_methods(file_path: str) -> List[Document]:
    """
    複数の方法でPDFファイルを読み込む関数

    Args:
        file_path (str): 読み込むPDFファイルのパス

    Returns:
        List[Document]: 読み込んだPDFファイルの内容を含むDocumentオブジェクトのリスト

    Raises:
        Exception: すべての方法が失敗した場合に発生する例外
    """
    # 使用可能なすべてのローダーとその設定のリスト
    loaders = [
        ("PDFPlumber", lambda: PDFPlumberLoader(file_path).load()),
        ("UnstructuredPDF-fast", lambda: UnstructuredPDFLoader(file_path, mode="elements", strategy="fast").load()),
        ("UnstructuredPDF-accurate", lambda: UnstructuredPDFLoader(file_path, mode="elements", strategy="accurate").load()),
        ("PDFMiner", lambda: PDFMinerLoader(file_path, laparams={"detect_vertical": True}).load()),
        ("PyPDF", lambda: PyPDFLoader(file_path).load()),
        # OCRを最後の手段として使用
        ("UnstructuredPDF-OCR", lambda: UnstructuredPDFLoader(
            file_path, 
            mode="elements", 
            strategy="ocr_only"
        ).load())
    ]

    last_error = None
    for loader_name, loader_func in loaders:
        try:
            doc = loader_func()
            print(f"成功 ({loader_name}): {file_path}")
            return doc
        except Exception as e:
            last_error = e
            print(f"{loader_name}が失敗: {file_path} - エラー: {str(e)}")
            continue
    
    # すべての方法が失敗した場合
    raise Exception(f"すべてのローダーが失敗: {last_error}")


def pdf_reader_run(file_paths: List[str]) -> List[Document]:
    """
    複数のPDFファイルを読み込み、その結果を返す

    Args:
        file_paths (List[str]): 読み込むPDFファイルのパスのリスト

    Returns:
        List[Document]: 読み込んだPDFファイルの内容を含むDocumentオブジェクトのリスト

    Raises:
        Exception: すべての方法が失敗した場合に発生する例外
    """
    # メイン処理
    docs = []
    failed_files = []

    for file_path in file_paths:
        try:
            doc = load_pdf_with_multiple_methods(file_path)
            docs.extend(doc)
        except Exception as e:
            failed_files.append((file_path, str(e)))
            print(f"すべての方法が失敗 {file_path}: {str(e)}")
            continue

    # 結果のサマリーを表示
    print("\n=== 処理結果サマリー ===")
    print(f"総ファイル数: {len(file_paths)}")
    print(f"成功したファイル数: {len(file_paths) - len(failed_files)}")
    print(f"失敗したファイル数: {len(failed_files)}")

    if failed_files:
        print("\n=== 失敗したファイル ===")
        for file_path, error in failed_files:
            print(f"ファイル: {file_path}")
            print(f"エラー: {error}")
            print("-" * 50)

    return docs