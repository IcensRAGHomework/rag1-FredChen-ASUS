from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import AzureChatOpenAI

from model_configurations import get_model_configuration

gpt_chat_version = "gpt-4o"
gpt_config = get_model_configuration(gpt_chat_version)


def get_model():
    return AzureChatOpenAI(
        model=gpt_config["model_name"],
        deployment_name=gpt_config["deployment_name"],
        openai_api_key=gpt_config["api_key"],
        openai_api_version=gpt_config["api_version"],
        azure_endpoint=gpt_config["api_base"],
        temperature=gpt_config["temperature"],
    )


def generate_hw01(question):
    model = get_model()
    examples = [
        {
            "input": "2024年台灣12月紀念日有哪些",
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
            "input": "昭和38年7月日本紀念日有哪",
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
                "You are an assistant that response user's question in json format.",
            ),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    chain = final_prompt | model
    response = chain.invoke({"input": question})
    return response.content


def generate_hw02(question):
    pass


def generate_hw03(question2, question3):
    pass


def generate_hw04(question):
    pass


def demo(question):
    llm = get_model()
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])

    return response
