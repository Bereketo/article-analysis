import os

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper


os.environ["AZURE_OPENAI_API_KEY"] = "1f5d1bb6920844248ea17f61f73f82ac"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-gpt-echo.openai.azure.com"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "o3-mini"

instructions = """You are a research assistant specialized in finding accurate and up-to-date information.
When searching for information, use specific search terms and analyze the results carefully.
Extract all possible relevant adverse/bad details from the search results to provide comprehensive answers.
Always cite your sources and provide context for the information you share.

# Notes:
- Parent company should be a search query only if it is having an impact on the company of interest.
- Add different variation of aliases in adverse_search_querires to make sure we are covering all the possible variations.
- Make sure the output is valide json without any comments or parsing issues.
- Always first adverse_search_querires should be company name + adverse news.

Output should be in json format.

Example output :
{
  "primary_alias": "Infosys",
  "aliases": ["Infosys Limited", "INFY", "Infosys India", "Infosys Ltd", "INFOS"],
  "stock_symbols": ["INFY"],
  "local_variants": ["Infosys India"],
  "parent_company": "Infosys",
  "adverse_search_querires": [
  "Infosys adverse news",
  "Infosys Limited scandal",
  "INFY fraud",
  "Infosys Ltd controversy",
  "Infosys India lawsuit",
  "INFOS whistleblower",
  "Infosys Limited SEC investigation",
  "INFY corporate governance issues",
  "Infosys Ltd accounting irregularities",
  "Infosys India regulatory violations",
  "INFOS compliance issues"
  ],
  "all_aliases" : "Infosys Limited, INFY, Infosys India, Infosys Ltd, INFOS"
}"""

base_prompt = hub.pull("langchain-ai/openai-functions-template")
api_wrapper =  BingSearchAPIWrapper(
    bing_subscription_key="822e2402879b4c78b00434c7f0f4c201",
    search_kwargs={"mkt": "en-IN"}
    # k=20
)
prompt = base_prompt.partial(instructions=instructions)
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    # temperature=0.0
)
tool = BingSearchResults(api_wrapper=api_wrapper, num_results=20)
tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
