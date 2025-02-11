from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=20)

def extract_company_name(query):
    template = """
    You are an entity extractor. Your role is to extract ONLY company names from questions.

    Instructions:
    - Extract ONLY single company name mentioned in the question
    - If no company name is present, respond with "分かりません"
    - If you are uncertain about the company name, respond with "分かりません"
    - If multiple company names are present (including merger/integration cases), respond with "分かりません"
    - Only provide the company name without any additional text

    Examples:
    Question: JR東日本の2023年度の営業収益は前年比でどのくらい増加しましたか？
    Answer: JR東日本

    Question: なぜ企業は環境に配慮した経営を重視するようになったのでしょうか？
    Answer: 分かりません

    Question: ソフトバンクグループの孫正義氏は何年に会長職に就任しましたか？
    Answer: ソフトバンクグループ

    Question: みずほ銀行と三井住友銀行の支店数を比較してください。
    Answer: 分かりません

    Question: イオンとセブン&アイの2022年度の売上高を比較してください。
    Answer: 分かりません

    Question: {input}
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(query)
    return response
