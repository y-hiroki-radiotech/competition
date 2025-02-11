from typing import List, Optional, Type
from pydantic import BaseModel, Field

# 企業概要モデル
class CompanyOverview(BaseModel):
    会社名: str = Field(..., description="会社名")
    本社所在地: str = Field(..., description="本社所在地")
    設立年月: str = Field(..., description="設立年月")
    代表者名: str = Field(..., description="代表者名")
    主要事業: str = Field(..., description="主要事業")
    上場市場: Optional[str] = Field(None, description="上場市場")
    証券コード: Optional[str] = Field(None, description="証券コード")
    URL: Optional[str] = Field(None, description="URL")

# 財務ハイライトモデル
class FinancialHighlights(BaseModel):
    売上高: int = Field(..., description="売上高")
    営業利益: int = Field(..., description="営業利益")
    経常利益: int = Field(..., description="経常利益")
    当期純利益: int = Field(..., description="当期純利益")
    総資産: int = Field(..., description="総資産")
    自己資本: int = Field(..., description="自己資本")
    ROE: float = Field(..., description="ROE")
    EPS: float = Field(..., description="EPS")

# ESGモデル
class ESG(BaseModel):
    環境: Optional[str] = Field(None, description="環境に関する取り組み")
    社会: Optional[str] = Field(None, description="社会に関する取り組み")
    ガバナンス: Optional[str] = Field(None, description="ガバナンスに関する取り組み")

# 株主情報モデル
class ShareholderInformation(BaseModel):
    株主数: int = Field(..., description="株主数")
    発行済株式数: int = Field(..., description="発行済株式数")
    配当金: float = Field(..., description="配当金")
    株価情報: Optional[str] = Field(None, description="株価情報")

# 経営戦略モデル
class ManagementStrategy(BaseModel):
    ビジョン: Optional[str] = Field(None, description="ビジョン")
    中期経営計画: Optional[str] = Field(None, description="中期経営計画")
    事業戦略: Optional[str] = Field(None, description="事業戦略")

# 組織情報モデル
class OrganizationInformation(BaseModel):
    組織図: Optional[str] = Field(None, description="組織図")
    従業員数: int = Field(..., description="従業員数")


def create_integrated_report_model(questionnaire_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    動的に生成されたQuestionnaireモデルを組み込んだIntegratedReportモデルを作成

    Args:
        questionnaire_model: 動的に生成されたQuestionnaireモデル

    Returns:
        カスタマイズされたIntegratedReportモデル
    """
    class CustomIntegratedReport(BaseModel):
        company_overview: CompanyOverview
        financial_highlights: FinancialHighlights
        esg: ESG
        shareholder_information: ShareholderInformation
        management_strategy: ManagementStrategy
        organization_information: OrganizationInformation
        other_information: Optional[str] = Field(None, description="その他の情報")
        questionnaire: questionnaire_model

    return CustomIntegratedReport


if __name__ == "__main__":
    from rag_code.create_pydantic_schema import CreatePydanticSchema
    from rag_code.create_integrated_pydantic_model import create_integrated_report_model

    search = CreatePydanticSchema("sample.pdf")
    question = "Pydanticのスキーマを生成してください。"
    schema_str = search.generate_pydantic_schema(question)
    print(schema_str)
    Questionnair = search.create_dynamic_pydantic_class(schema_str)
    IntegratedReport = create_integrated_report_model(Questionnair)
