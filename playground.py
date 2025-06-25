import openai
from phidata.agent import Agent
import phidata.api_key
from phidata.model.openai import OpenAIChat
from phidata.tools.yfinance import YFinanceTools
from phidata.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phidata.model.groq import Groq

import os
import phidata
from phidata.playground import Playground, serve_playgraound_app
#Load enviornment variables from .env file
load_dotenv()

phidata.api=os.getenv("phidata_API_KEY")

##web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="Llama-3.3-70b-Versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,

)

## financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="Llama-3.3-70b-Versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,


)

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playgraound_app("playground:app",reload=True)