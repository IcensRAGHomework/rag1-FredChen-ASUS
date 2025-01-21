import os

import requests
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel

from model_configurations import get_model_configuration

gpt_chat_version = "gpt-4o"
gpt_config = get_model_configuration(gpt_chat_version)


class Holiday(BaseModel):
    date: str
    name: str


class HolidayResponse(BaseModel):
    Result: list[Holiday]


def get_chat_model():
    return AzureChatOpenAI(
        model=gpt_config["model_name"],
        deployment_name=gpt_config["deployment_name"],
        openai_api_key=gpt_config["api_key"],
        openai_api_version=gpt_config["api_version"],
        azure_endpoint=gpt_config["api_base"],
        temperature=gpt_config["temperature"],
    )


@tool
def get_holidays_by_country_code_and_year(
    country_code="TW", year=2024
) -> list[dict[str, str]]:
    """Get holiday list by country code and year
    Args:
        country_code (str): Country code in ISO 3166-1 alpha-2 format
        year (int): Year in YYYY format
    Returns:
        list[dict[str, str]]: List of holidays with date and name
    """
    api_key = os.environ.get("CALENDARIFIC_API_KEY")
    response = requests.get(
        f"https://calendarific.com/api/v2/holidays",
        params={"api_key": api_key, "country": country_code, "year": year},
    )
    response.raise_for_status()
    holidays_full_list = response.json().get("response", {}).get("holidays", [])
    holidays_list = []
    for item in holidays_full_list:
        clean_item = {
            "date": item.get("date").get("iso"),
            "name": item.get("name"),
        }
        holidays_list.append(clean_item)
    return holidays_list


def get_holiday_list_prompt():
    examples = [
        {
            "input": "2024年台灣10月紀念日有哪些",
            "output": {"Result": [{"date": "2024-10-10", "name": "國慶日"}]},
        },
        {
            "input": "2025第一個月的紀念日有哪些",
            "output": {
                "Result": [
                    {"date": "2025-01-01", "name": "元旦"},
                    {"date": "2025-01-21", "name": "除夕"},
                ]
            },
        },
        {
            "input": "昭和38年7月日本紀念日有哪些",
            "output": {"Result": [{"date": "1963-07-01", "name": "海の日"}]},
        },
    ]
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your job is to help users find holidays in certain time period",
            ),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt


def generate_hw01(question):
    model = get_chat_model()
    prompt_template = get_holiday_list_prompt()
    chain = prompt_template | model.with_structured_output(HolidayResponse)
    result: HolidayResponse = chain.invoke({"input": question})
    return result.model_dump_json()


def generate_hw02(question):
    model = get_chat_model()
    tools = [get_holidays_by_country_code_and_year]

    def _modify_state_messages(state: AgentState):
        prompt_template = get_holiday_list_prompt()
        return prompt_template.invoke({"input": state["messages"]}).to_messages()

    app = create_react_agent(
        model,
        tools,
        state_modifier=_modify_state_messages,
        response_format=HolidayResponse,
    )

    result = app.invoke({"messages": [("human", question)]})
    return result.model_dump_json()


def generate_hw03(question2, question3):
    pass


def generate_hw04(question):
    pass


def demo(question):
    llm = get_chat_model()
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])

    return response
