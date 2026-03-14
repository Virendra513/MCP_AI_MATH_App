import streamlit as st
import asyncio
import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# MCP Servers
SERVERS = {
    "Math Server": {
        "transport": "streamable_http",
        "url": "https://math-mcp.fastmcp.app/mcp"
    },
    "Fetch Server": {
        "transport": "streamable_http",
        "url": "https://remote.mcpservers.org/fetch/mcp"
    }
}

# Convert LangChain tools → OpenAI format
def convert_to_openai_tools(langchain_tools):
    openai_tools = []

    for tool in langchain_tools:

        parameters = {"type": "object", "properties": {}}

        if hasattr(tool, "args_schema") and tool.args_schema:
            if isinstance(tool.args_schema, dict):
                parameters = tool.args_schema
            else:
                try:
                    parameters = tool.args_schema.schema()
                except:
                    pass

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters
            }
        })

    return openai_tools


# Initialize MCP + Model
@st.cache_resource
def init_agent():

    async def setup():
        client = MultiServerMCPClient(SERVERS)
        tools = await client.get_tools()
        return client, tools

    client, langchain_tools = asyncio.run(setup())

    named_tools = {tool.name: tool for tool in langchain_tools}
    openai_tools = convert_to_openai_tools(langchain_tools)

    hf_client = InferenceClient(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    return hf_client, named_tools, openai_tools


hf_client, named_tools, tools = init_agent()


# Run MCP agent
async def run_agent(messages):

    response = hf_client.chat.completions.create(
        model="openai/gpt-oss-120b:novita",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if not getattr(message, "tool_calls", None):
        return message.content

    tool_messages = []

    for tc in message.tool_calls:

        tool_name = tc.function.name
        args = json.loads(tc.function.arguments)

        result = await named_tools[tool_name].ainvoke(args)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result)
        })

    messages.append(message)
    messages.extend(tool_messages)

    final = hf_client.chat.completions.create(
        model="openai/gpt-oss-120b:novita",
        messages=messages
    )

    return final.choices[0].message.content


# Streamlit UI
st.set_page_config(page_title="MCP AI Chatbot", page_icon="🤖")

st.title("🤖 MCP AI Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User input
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = asyncio.run(run_agent(st.session_state.messages.copy()))

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )