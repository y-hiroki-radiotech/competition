from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
import openai
import json

load_dotenv()

class CreatePydanticSchema:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = self.filepath.split(".")[-2].split("/")[-1]

    def create_pydantic_schema_prompt(self, question: str) -> str:
        """"GPT-4に送信するプロンプトを生成する"""
        return f"""
    以下の質問から、的確な質問のポイントを押さえたPydanticのスキーマを生成してください。
    必要に応じて、関連する追加フィールドも提案してください。
    各フィールドには適切なデータ型とdescriptionをつけてください。
    以下は出力の例です:

    入力例:
    日産自動車の2023年度の従業員の平均年収は約何万円でしょうか。

    出力例：
    {{
        "fields": [
            {{
                "name": "従業員平均年収",
                "type": "int",
                "description": "2023年度の日産自動車従業員の平均年収（単位：万円）",
            }},
        ]
    }}
    あなたの質問:
    {question}

    出力形式:
    {{
        "fields": [
            {{
                "name": "フィールド名",
                "type": "データ型",
                "description": "説明",
                "optional": true/false
            }},
            ...
        ]
    }}

    注意事項:
    - 数値データは適切な型(int, float)を選択してください
    - 必須/オプションを適切に設定してください
    - descriptionは具体的で明確な説明を付けてください
    - 単位が必要な場合は、descriptionに明記してください
    """

    def generate_pydantic_schema(self, question: str) -> Dict[str, Any]:
        """GPT-4を使用してPydanticスキーマを生成する"""
        client = openai.OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたはPydanticスキーマの専門家です。"},
                {"role": "user", "content": self.create_pydantic_schema_prompt(question)},
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    def parse_schema_response(self, schema_str: str) -> Dict[str, Any]:
        """
        GPT-4からの応答をパースしてPythonの辞書に変換します。

        Args:
            schema_str (str): GPT-4から返された JSON 形式の文字列

        Returns:
            Dict[str, Any]: パースされたスキーマ辞書
        """
        try:
            # コードブロックのマーカー（```json や ```）を削除
            cleaned_str = schema_str.replace('```json', '').replace('```', '').strip()

            # JSON文字列をパース
            schema_dict = json.loads(cleaned_str)

            return schema_dict
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing schema: {e}")

    def create_dynamic_pydantic_class(self, schema_str: str, class_name: str = "Questionnaire") -> type[BaseModel]:
        schema_dict = self.parse_schema_response(schema_str)

        field_definitions = {}

        for field in schema_dict["fields"]:
            field_type = eval(field["type"])

            if field["optional"]:
                field_type = Optional[field_type]

            # OBJECT型でpropertiesが空の場合の対処
            if field["type"] == "Dict[str, int]" or field["type"] == "Dict[str, str]" or field["type"] == "Dict[str, Any]": #Dict型の場合
                if "properties" not in field or not field["properties"]:
                    # propertiesが空の場合は、Any型を指定する
                    field_type = Dict[str, Any] #変更
                    print("field_typeを変更")

            field_definitions[field["name"]] = (
                field_type,
                Field(... if not field["optional"] else None,
                    description=field["description"]
                    )
                )
        return create_model(class_name, **field_definitions)



if __name__ == "__main__":
    search = CreatePydanticSchema("sample.pdf")
    question = "Pydanticのスキーマを生成してください。"
    schema_str = search.generate_pydantic_schema(question)
    print(schema_str)
    Questionnair = search.create_dynamic_pydantic_class(schema_str)
