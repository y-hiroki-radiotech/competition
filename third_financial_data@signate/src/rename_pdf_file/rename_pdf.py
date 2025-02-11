import logging
import os
import re
from typing import List, Optional

import pikepdf
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm
import pdfplumber  # より良い文字エンコーディングサポート
import re
import mojimoji

load_dotenv()
logging.basicConfig(level=logging.INFO, force=True)


class PdfRename:
    """
    PDFファイルの名前を変更し、メタデータを更新するためのクラス。

    このクラスは以下の機能を提供します：
    - PDFファイルのテキスト抽出（OCR処理を含む）
    - OpenAIを使用したタイトル生成
    - PDFファイルの名前変更
    - メタデータの更新

    Attributes:
        llm (OpenAI): OpenAI APIクライアントインスタンス
        file_path (str): 処理対象のPDFファイルが存在するディレクトリパス
        pdf_files (List[str]): 処理対象のPDFファイルパスのリスト
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        """
        PdfRenameクラスの初期化。

        Args:
            file_path (Optional[str]): PDFファイルが存在するディレクトリパス。
                                     Noneの場合、デフォルトのパスが使用されます。

        Raises:
            FileNotFoundError: 指定されたディレクトリが存在しない場合
        """
        self.llm = OpenAI()
        self.file_path = file_path or self._get_default_file_path()
        self.pdf_files = self._get_pdf_files()

    def _get_default_file_path(self) -> str:
        """
        デフォルトのファイルパスを取得する。

        Returns:
            str: デフォルトのファイルパス（現在のディレクトリの'documents'フォルダ）

        Raises:
            FileNotFoundError: デフォルトのファイルパスが存在しない場合
        """
        current_dir = os.getcwd()
        default_path = os.path.join(current_dir, "documents")
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"デフォルトのファイルパスが見つかりません: {default_path}・・")
        return default_path

    def _generate_title(self, text: str) -> str:
        """
        テキストから会社名と報告書の種類を抽出してタイトルを生成する。

        Args:
            text (str): 解析対象のテキスト

        Returns:
            str: 生成されたタイトル（形式: 会社名-報告書の種類-年）
        """
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """あなたは優秀な日本の経済アナリストです。以下のテキストから会社名と報告書の種類を抽出してください。\n
                    会社名が複数ある場合は、その中のホールディング会社を会社名としてください。\n
                    報告書の種類は、与えられたテキストの初めに書いてあることが多いです。""",
                },
                {"role": "user", "content": text},
                {"role": "system", "content": "以下の形式で出力してください: 会社名-報告書の種類-年"},
            ],
        )
        return response.choices[0].message.content.strip()

    def _clean_text(self, text: str) -> str:
        """
        抽出したテキストのクリーニングを行う。

        Args:
            text (str): クリーニング対象のテキスト

        Returns:
            str: クリーニング済みのテキスト
        """
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_text_from_image(self, file_path: str) -> str:
        """
        PDFをOCR処理してテキストを抽出する。

        Args:
            file_path (str): 処理対象のPDFファイルパス

        Returns:
            str: 抽出されたテキスト（最初のページのみ）
        """
        images = convert_from_path(file_path)
        text_content = []
        # タイトルを抜き出すので、最初のページだけ処理する
        for image in images[:1]:
            text = pytesseract.image_to_string(image, lang="jpn")
            text_content.append(text)
            break
        return text_content[0]

    def _rename_pdf_ocr(self, file_path: str) -> None:
        """
        PDFファイルの名前を変更し、メタデータを更新する。

        Args:
            file_path (str): 処理対象のPDFファイルパス
        """
        try:
            # OCRでテキストを抽出
            first_page_text = self._extract_text_from_image(file_path)
            title = self._generate_title(first_page_text)

            # 新しいファイル名を生成
            output_path = os.path.join(os.path.dirname(file_path), f"{title}.pdf")

            # pikepdfを使用してメタデータを更新
            with pikepdf.open(file_path,  allow_overwriting_input=True) as pdf:
                with pdf.open_metadata() as meta:
                    meta["dc:title"] = title
                pdf.save(output_path)

            # 元のファイルを削除
            os.remove(file_path)
            logging.info(f"Renamed and updated metadata: {file_path} -> {output_path}")

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            raise

    def _clean_text(self, text):
        text = re.sub(r'([0-9])\s+([0-9])', r'\1\2', text)
        # 全角を半角に変換 (数字、アルファベット、スペース)
        text = mojimoji.zen_to_han(text, kana=False)
        # 改行をスペースに変換
        text = text.replace('\n', ' ')
        # (cid:xx)形式の文字を削除
        text = re.sub(r'\(cid:[0-9]+\)', '', text)
        # 連続するスペースを1つに
        text = re.sub(r'\s+', ' ', text)
        # 数字間の不要なスペースを削除 (例: 2 0 2 4 → 2024)
        text = re.sub(r'([0-9])\s+([0-9])', r'\1\2', text)

        return text.strip()

    def _extract_first_last_page(self, file_path: str):
        try:
            # pdfplumberを使用してPDFを開く
            with pdfplumber.open(file_path) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                # 最後から3ページ分取得
                text = pdf.pages[-3].extract_text()
                text = pdf.pages[-2].extract_text()
                text += pdf.pages[-1].extract_text()
                return self._clean_text(text)
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            return None

    def _rename_pdf(self, file_path: str) -> None:
        """
        PDFファイルの名前を変更し、メタデータを更新する。

        Args:
            file_path (str): 処理対象のPDFファイルパス
        """
        try:
            # テキストを抽出
            first_last_text = self._extract_first_last_page(file_path)
            title = self._generate_title(first_last_text)

            # 新しいファイル名を生成
            output_path = os.path.join(os.path.dirname(file_path), f"{title}.pdf")

            # pikepdfを使用してメタデータを更新
            with pikepdf.open(file_path,  allow_overwriting_input=True) as pdf:
                with pdf.open_metadata() as meta:
                    meta["dc:title"] = title
                pdf.save(output_path)

            # 元のファイルを削除
            os.remove(file_path)
            logging.info(f"Renamed and updated metadata: {file_path} -> {output_path}")

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            raise

    def _get_pdf_files(self) -> List[str]:
        """
        指定されたディレクトリからPDFファイルのリストを取得する。

        Returns:
            List[str]: PDFファイルパスのリスト
        """
        pdf_files = []
        for file in os.listdir(self.file_path):
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(self.file_path, file))
        return pdf_files

    def rename_pdfs(self, ocr=False) -> None:
        """
        すべてのPDFファイルの名前を変更する。

        Note:
            このメソッドは self.pdf_files 内のすべてのPDFファイルを処理します。
        """
        if not ocr:
            for pdf_file in tqdm(self.pdf_files, total=len(self.pdf_files), position=0):
                self._rename_pdf(pdf_file)
        else:
            for pdf_file in tqdm(self.pdf_files, total=len(self.pdf_files), position=0):
                self._rename_pdf_ocr(pdf_file)




if __name__ == "__main__":
    pdf_rename = PdfRename()
    pdf_rename.rename_pdfs()
    # 繰り返し使う時はもう一度コンストラクトする
    pdf_rename = PdfRename()
    pdf_rename.rename_pdfs(ocr=True)
