from google import genai
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
import openai
import json



def response_gemini(question: str, IntegratedReport) -> Dict[str, Any]:
    load_dotenv()
    client = genai.Client()
    model_id = "gemini-2.0-flash"
    prompt = "以下のPDFファイルをGemini Flash APIで読み取り、Pydanticモデルに基づいて構造化されたJSON形式で出力してください。"
    response = client.models.generate_content(model=model_id, contents=prompt, config={"response_mime_type": "application/json", "response_schema": IntegratedReport})
    return response.text


def create_context(model_fields, data_str):
    """
    Pydanticモデルのフィールド情報とデータを組み合わせてコンテキストを生成する

    Args:
        model_fields (dict): Pydanticモデルのフィールド情報
        data_str (str): JSON形式の文字列データ

    Returns:
        str: 生成されたコンテキスト
    """
    # JSON文字列をパース
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # クエスチョネアデータの抽出
    questionnaire_data = data.get('questionnaire', {})

    # コンテキストの構築
    context_parts = []

    # 各フィールドについて処理
    for field_name, field_info in model_fields.items():
        if field_name in questionnaire_data:
            value = questionnaire_data[field_name]
            description = field_info.description
            context_parts.append(f"{description}: {value}")

    # コンテキストの結合
    context = "コンテキスト：\n" + "\n".join(context_parts)

    return context


if __name__ == "__main__":
    from rag_code.create_pydantic_schema import CreatePydanticSchema
    from rag_code.create_integrated_pydantic_model import create_integrated_report_model
    from rag_code.gemini_flash_pdf_output import response_gemini
    from rag_code.gemini_flash_pdf_output import create_context

    search = CreatePydanticSchema("sample.pdf")
    question = "Pydanticのスキーマを生成してください。"
    schema_str = search.generate_pydantic_schema(question)
    print(schema_str)
    Questionnair = search.create_dynamic_pydantic_class(schema_str)
    IntegratedReport = create_integrated_report_model(Questionnair)
    response = response_gemini(question, IntegratedReport)
    context = create_context(Questionnair.model_fields, response)
