import os

import requests
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
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
def get_holidays_by_country_code_and_year_and_month(
    country_code="TW", year=2024, month=10
) -> list[dict[str, str]]:
    """Get holiday list by country code, year, and month
    Args:
        country_code (str): Country code in ISO 3166-1 alpha-2 format
        year (int): Year in YYYY format
        month (int): Month in MM format
    Returns:
        list[dict[str, str]]: List of holidays with date and name
    """
    api_key = os.environ.get("CALENDARIFIC_API_KEY")

    # Mock data if no API key provided
    holidays_full_list = []
    if not api_key:
        holidays_full_list = [
            {"date": "2024-01-01", "name": "Republic Day/New Year's Day"},
            {"date": "2024-02-04", "name": "Farmer's Day"},
            {"date": "2024-02-08", "name": "Lunar New Year Holiday"},
            {"date": "2024-02-09", "name": "Lunar New Year's Eve"},
            {"date": "2024-02-10", "name": "Lunar New Year's Day"},
            {"date": "2024-02-11", "name": "Lunar New Year Holiday"},
            {"date": "2024-02-12", "name": "Lunar New Year Holiday"},
            {"date": "2024-02-13", "name": "Lunar New Year Holiday"},
            {"date": "2024-02-14", "name": "Lunar New Year Holiday"},
            {"date": "2024-02-17", "name": "Special Working Day"},
            {"date": "2024-02-24", "name": "Lantern Festival"},
            {"date": "2024-02-24", "name": "Tourism Day"},
            {"date": "2024-02-28", "name": "Peace Memorial Day"},
            {"date": "2024-03-08", "name": "International Women's Day"},
            {"date": "2024-03-11", "name": "Earth God's Birthday"},
            {"date": "2024-03-12", "name": "Arbor Day"},
            {"date": "2024-03-20T11:06:28+08:00", "name": "March Equinox"},
            {"date": "2024-03-28", "name": "Kuan Yin's Birthday"},
            {"date": "2024-03-29", "name": "Youth Day"},
            {"date": "2024-03-31", "name": "Easter Sunday"},
            {"date": "2024-04-04", "name": "Tomb Sweeping Day"},
            {"date": "2024-04-04", "name": "Children's Day"},
            {"date": "2024-04-05", "name": "Children's Day/Tomb Sweeping Day Holiday"},
            {"date": "2024-04-06", "name": "Children's Day/Tomb Sweeping Day Holiday"},
            {"date": "2024-04-07", "name": "Children's Day/Tomb Sweeping Day Holiday"},
            {"date": "2024-04-23", "name": "God of Medicine's Birthday"},
            {"date": "2024-05-01", "name": "Labor Day"},
            {"date": "2024-05-01", "name": "Matsu's Birthday"},
            {"date": "2024-05-04", "name": "Literary Day"},
            {"date": "2024-05-12", "name": "Mother's Day"},
            {"date": "2024-05-15", "name": "Buddha's Birthday"},
            {"date": "2024-06-03", "name": "Opium Suppression Movement Day"},
            {"date": "2024-06-10", "name": "Dragon Boat Festival"},
            {"date": "2024-06-18", "name": "Kuan Kung's Birthday"},
            {"date": "2024-06-18", "name": "Chen Huang's Birthday"},
            {"date": "2024-06-21T04:50:59+08:00", "name": "June Solstice"},
            {"date": "2024-08-08", "name": "Father's Day"},
            {"date": "2024-08-10", "name": "Chinese Valentine's Day"},
            {"date": "2024-08-18", "name": "Hungry Ghost Festival"},
            {"date": "2024-09-03", "name": "Armed Forces Day"},
            {"date": "2024-09-17", "name": "Mid-Autumn Festival"},
            {"date": "2024-09-22T20:43:33+08:00", "name": "September Equinox"},
            {"date": "2024-09-28", "name": "Teachers' Day"},
            {"date": "2024-10-10", "name": "National Day"},
            {"date": "2024-10-11", "name": "Double Ninth Day"},
            {"date": "2024-10-21", "name": "Overseas Chinese Day"},
            {"date": "2024-10-25", "name": "Taiwan's Retrocession Day"},
            {"date": "2024-10-31", "name": "Halloween"},
            {"date": "2024-11-12", "name": "Sun Yat-sen's Birthday"},
            {"date": "2024-11-15", "name": "Saisiat Festival"},
            {"date": "2024-12-21T17:20:34+08:00", "name": "December Solstice"},
            {"date": "2024-12-21T17:20:34+08:00", "name": "Dōngzhì Festival"},
            {"date": "2024-12-25", "name": "Constitution Day"},
            {"date": "2024-12-25", "name": "Christmas Day"},
        ]
    else:
        response = requests.get(
            f"https://calendarific.com/api/v2/holidays",
            params={"api_key": api_key, "country": country_code, "year": year},
        )
        response.raise_for_status()
        holidays_full_list = response.json().get("response", {}).get("holidays", [])
    holidays_list = []
    for item in holidays_full_list:
        clean_item = {
            # some date will contain time, drop it
            "date": item.get("date").get("iso")[:10],
            "name": item.get("name"),
        }
        if int(clean_item["date"][5:7]) == month:
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
    tools = [get_holidays_by_country_code_and_year_and_month]

    def _modify_state_messages(state: AgentState):
        prompt_template = get_holiday_list_prompt()
        return prompt_template.invoke({"input": state["messages"]}).to_messages()

    app = create_react_agent(
        model,
        tools,
        state_modifier=_modify_state_messages,
        response_format=HolidayResponse,
    )

    result = app.invoke(
        {"messages": [("human", question)]},
        stream_mode="updates",
    )
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
