import os
from typing import Optional, Tuple, List
import fitz
import pymupdf4llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class PDFPreprocessor:
    def __init__(self, pdf_directory: str):
        """
        初期化関数

        Args:
            pdf_directory (str): PDFファイルが格納されているディレクトリパス
        """
        self.pdf_directory = pdf_directory
        self.company_to_file_info = {}  # 会社名から[ファイルパス, 元のファイル名]へのマッピング
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # PDFファイルの登録
        self._register_pdfs()

    def _register_pdfs(self) -> None:
        """ディレクトリ内のPDFファイルを登録"""
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith(".pdf"):
                company_name = self._extract_company_name(filename)
                filepath = os.path.join(self.pdf_directory, filename)
                self.company_to_file_info[company_name] = (filepath, filename)
                print(f"登録: {company_name} (ファイル名: {filename})")

    def _extract_company_name(self, filename: str) -> str:
        """ファイル名から会社名を抽出"""
        return filename.split("-")[0]

    def _identify_company_with_llm(self, query: str, available_companies: List[str]) -> Optional[str]:
        """
        LLMを使用してクエリから会社名を特定

        Args:
            query (str): 検索クエリ
            available_companies (List[str]): 利用可能な会社名のリスト

        Returns:
            Optional[str]: 特定された会社名。見つからない場合はNone
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            あなたは日本の企業を特定する専門家です。
            ユーザーのクエリから、提供された企業リストの中から最も関連する企業を1つ特定してください。

            以下の点に注意してください：
            - クエリに企業名が明示的に含まれている場合、その企業を選択
            - クエリに企業名が明示的に含まれていない場合、文脈から最も関連する企業を推測
            - 複数の企業が候補となる場合、最も関連が強いものを1つ選択
            - 適切な企業が見つからない場合は "該当なし" と回答

            """),
            ("human", """
            クエリ: {query}

            利用可能な企業リスト:
            {companies}
            """)
        ])

        # LLMで会社を特定
        companies_str = "\n".join(available_companies)
        response = prompt | self.llm | StrOutputParser()
        company_name = response.invoke({
            "query": query,
            "companies": companies_str
        }).strip()

        if company_name == "該当なし":
            return None

        return company_name

    def _convert_to_markdown(self, pdf_path: str) -> Optional[str]:
        """PDFをMarkdownに変換"""
        try:
            # まずpymupdf4llmで試行
            print(f"pymupdf4llmで変換を試みます: {pdf_path}")
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            print("pymupdf4llmでの変換が成功しました")
            return markdown_text
        except Exception as e:
            print(f"pymupdf4llmでの変換に失敗しました: {str(e)}")
            print("fitzを使用して再試行します...")

            try:
                # fitzにフォールバック
                doc = fitz.open(pdf_path)
                markdown_text = ""
                for page in doc:
                    text = page.get_text()
                    markdown_text += text + "\n\n"
                doc.close()

                # 構造化
                markdown_lines = markdown_text.splitlines()
                structured_text = ""
                paragraph = ""

                for line in markdown_lines:
                    line = line.strip()
                    if line:
                        paragraph += line + " "
                    else:
                        if paragraph:
                            structured_text += paragraph.strip() + "\n\n"
                            paragraph = ""

                if paragraph:
                    structured_text += paragraph.strip() + "\n\n"

                print("fitzでの変換が成功しました")
                return structured_text

            except Exception as e:
                print(f"fitzでの変換にも失敗しました: {str(e)}")
                return None

    def _save_markdown(self, markdown_text: str, pdf_path: str) -> Optional[str]:
        """Markdownテキストをファイルとして保存"""
        try:
            md_path = pdf_path.rsplit(".", 1)[0] + ".md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            print(f"Markdownファイルを保存しました: {md_path}")
            return md_path
        except Exception as e:
            print(f"Markdownファイルの保存に失敗しました: {str(e)}")
            return None

    def find_company_and_convert(self, query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        クエリから会社を特定し、PDFをMarkdownに変換

        Args:
            query (str): 検索クエリ

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]:
            (会社名, Markdownファイルパス, 元のPDFファイル名)のタプル
        """
        # 利用可能な会社のリストを取得
        available_companies = list(self.company_to_file_info.keys())

        # LLMで会社を特定
        company_name = self._identify_company_with_llm(query, available_companies)

        if not company_name:
            print("該当する会社が見つかりませんでした")
            return None, None, None

        print(f"対象企業: {company_name}")

        # ファイル情報の取得
        if company_name not in self.company_to_file_info:
            print(f"エラー: {company_name} のファイル情報が見つかりません")
            return None, None, None

        pdf_path, original_filename = self.company_to_file_info[company_name]

        # PDFが存在するか確認
        if not os.path.exists(pdf_path):
            print(f"エラー: PDFファイルが見つかりません: {pdf_path}")
            return None, None, None

        # Markdownファイルのパスを取得
        md_path = pdf_path.rsplit(".", 1)[0] + ".md"

        # Markdownファイルが既に存在する場合はそれを使用
        if os.path.exists(md_path):
            print(f"既存のMarkdownファイルを使用します: {md_path}")
            return company_name, md_path, original_filename

        # PDFをMarkdownに変換
        markdown_text = self._convert_to_markdown(pdf_path)
        if markdown_text is None:
            print("PDF変換に失敗しました")
            return None, None, None

        # Markdownを保存
        md_path = self._save_markdown(markdown_text, pdf_path)
        if md_path is None:
            print("Markdownファイルの保存に失敗しました")
            return None, None, None

        return company_name, md_path, original_filename

# 使用例
if __name__ == "__main__":
    from preprocess.pdf2markdown import PDFPreprocessor
    # PDFファイルのディレクトリを指定
    pdf_dir = "/Users/user/Desktop/GenerativeAI_apps/third_finance_competition/validation/documents"

    # プリプロセッサーを初期化
    preprocessor = PDFPreprocessor(pdf_dir)

    # クエリから会社を特定してPDFを変換
    query = "ダイドーグループの2030年のビジョンについて教えてください"
    company_name, md_path, original_filename = preprocessor.find_company_and_convert(query)

    if company_name and md_path:
        print(f"処理が完了しました:")
        print(f"- 会社名: {company_name}")
        print(f"- Markdownファイル: {md_path}")
        print(f"- 元のPDFファイル: {original_filename}")
