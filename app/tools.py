# =====================================================
# app/tools.py
# Production Tool Layer
# Exposes all tools available to the LangGraph agent.
# =====================================================

import logging
import requests

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from app.config import ALPHA_VANTAGE_API_KEY
from app.rag import retrieve_context

# -----------------------------------------------------
# Logger
# -----------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Web Search Tool
# -----------------------------------------------------

_search = DuckDuckGoSearchRun(region="us-en")


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    """
    return _search.run(query)


# -----------------------------------------------------
# Calculator Tool
# -----------------------------------------------------

@tool
def calculator(
    a: float,
    b: float,
    operation: str
) -> str:
    """
    Perform basic arithmetic.

    Supported operations:
    add, sub, mul, div
    """

    try:
        if operation == "add":
            result = a + b

        elif operation == "sub":
            result = a - b

        elif operation == "mul":
            result = a * b

        elif operation == "div":
            if b == 0:
                return "Error: Division by zero."

            result = a / b

        else:
            return "Error: Invalid operation."

        return f"Result: {result}"

    except Exception as e:
        logger.exception("Calculator tool failed")
        return f"Calculator error: {str(e)}"


# -----------------------------------------------------
# Stock Price Tool
# -----------------------------------------------------

@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch the latest stock price using Alpha Vantage.
    """

    try:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol.upper()}"
            f"&apikey={ALPHA_VANTAGE_API_KEY}"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        quote = data.get("Global Quote", {})

        if not quote:
            return f"No stock data found for {symbol.upper()}."

        return (
            f"{symbol.upper()} is trading at "
            f"${quote['05. price']} today. "
            f"Change: {quote['09. change']} "
            f"({quote['10. change percent']}) "
            f"as of {quote['07. latest trading day']}."
        )

    except requests.RequestException:
        logger.exception("Stock API request failed")
        return "Unable to retrieve stock data at the moment."


# -----------------------------------------------------
# RAG Retrieval Tool
# -----------------------------------------------------

@tool
def retrieve_documents(query: str) -> str:
    """
    Search uploaded PDF documents and return
    the most relevant context with citations.
    """

    logger.info("RAG CALLED: %s", query)

    contexts = retrieve_context(query)

    if not contexts:
        return "No relevant information found in uploaded documents."

    output = []

    for item in contexts:
        output.append(
            f"""Source: {item["source"]}
Page: {item["page"]}

Content:
{item["text"]}"""
        )

    return "\n\n".join(output)


# -----------------------------------------------------
# Export Tool Registry
# -----------------------------------------------------

tools = [
    web_search,
    calculator,
    get_stock_price,
    retrieve_documents,
]