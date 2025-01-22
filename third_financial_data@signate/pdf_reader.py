import re
import unicodedata
from typing import List

from langchain_community.document_loaders import (PDFMinerLoader,
                                                  PDFPlumberLoader,
                                                  PyPDFLoader,
                                                  UnstructuredPDFLoader)


def load_pdf_with_multiple_methods(file_path: str) -> List[str]:
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
        ("UnstructuredPDF-hi_res", lambda: UnstructuredPDFLoader(file_path, mode="elements", strategy="hi_res").load()),
        ("UnstructuredPDF-fast", lambda: UnstructuredPDFLoader(file_path, mode="elements", strategy="fast").load()),
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


def pdf_reader_run(file_paths: List[str]) -> List[str]:
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
            raw_docs = load_pdf_with_multiple_methods(file_path)
            cleaned_docs = [clean_document(doc) for doc in raw_docs]
            docs.extend(cleaned_docs)
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



def clean_text(text: str) -> str:
    """
    テキストを整形する関数
    
    Args:
        text (str): 整形する生のテキスト
    Returns:
        str: 整形されたテキスト
    """
    # 基本的なクリーニング
    cleaned = text.strip()
        
    # 特殊文字の正規化
    cleaned = unicodedata.normalize('NFKC', cleaned)

    
    # PDFから抽出した際によく見られる問題に対する処理
    cleaned = re.sub(r'\n+', '\n', cleaned)  # 複数の改行を1つに
    cleaned = re.sub(r'([。．！？])\s*\n', r'\1\n', cleaned)  # 文末の改行を整理
    cleaned = re.sub(r'([。．！？」］｝】〕〉》』】）}])', r'\1\n', cleaned)  # 句点での改行追加
    
    # 半角カタカナを全角に変換
    cleaned = ''.join(chr(0xFF00 + (ord(ch) - 0x20)) if 0x20 <= ord(ch) <= 0x7E else ch for ch in cleaned)
    
    # 余分な空白の処理
    cleaned = re.sub(r'\s+', ' ', cleaned)  # 連続する空白を1つに
    cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)  # 各行の先頭と末尾の空白を削除
    cleaned = re.sub(r'([0-9])\s+([0-9])', r'\1\2', cleaned)  # 数字間の不要な空白を削除
    
    # 日付や数値の形式を整える
    cleaned = re.sub(r'([0-9])\s*(年|月|日|円|個|件|人|時|分|秒)', r'\1\2', cleaned)
    
    # 不要な制御文字の削除
    cleaned = ''.join(char for char in cleaned if not unicodedata.category(char).startswith('C'))
    
    # 全角スペースを半角に変換
    cleaned = cleaned.replace('　', ' ')

    cleaned = unicodedata.normalize('NFKC', cleaned)

    return cleaned

def fix_mojibake(text: str) -> str:
    """
    文字化けを修正する関数
    
    Args:
        text (str): 修正する文字列
        
    Returns:
        str: 修正された文字列
    """
    replacements = {
        'ï¼': '：',
        'â': '―',
        '™': '㈱',
        'Ⅰ': '1',
        'Ⅱ': '2',
        'Ⅲ': '3',
        'Ⅳ': '4',
        'Ⅴ': '5',
        # 必要に応じて追加
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def clean_document(doc):
    """
    Documentオブジェクトのテキストを整形する関数
    
    Args:
        doc: Documentオブジェクト
        
    Returns:
        整形されたDocumentオブジェクト
    """
    # 文字化け修正を適用
    doc.page_content = fix_mojibake(doc.page_content)
    # テキストクリーニングを適用
    doc.page_content = clean_text(doc.page_content)
    return doc