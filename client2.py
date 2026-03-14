import asyncio
import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage

load_dotenv()

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

async def main():

    mcp_client = MultiServerMCPClient(SERVERS)
    #tools = await mcp_client.get_tools()
    langchain_tools = await mcp_client.get_tools()

    #named_tools = {tool.name: tool for tool in tools}
    named_tools = {tool.name: tool for tool in langchain_tools}
    tools = convert_to_openai_tools(langchain_tools)

    print("Available tools:", named_tools.keys())

    hf_client = InferenceClient(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    prompt = " fetch this https://virendra513.github.io/mywebsite/ and give rsponse in 20 words"

    messages = [{"role": "user", "content": prompt}]

    response = hf_client.chat.completions.create(
        model="openai/gpt-oss-120b:novita",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # If no tool needed
    if not getattr(message, "tool_calls", None):
        print("LLM Reply:", message.content)
        return

    tool_messages = []

    for tc in message.tool_calls:
        tool_name = tc.function.name
        args = json.loads(tc.function.arguments)
        tool_id = tc.id

        result = await named_tools[tool_name].ainvoke(args)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": json.dumps(result)
        })

    messages.append(message)
    messages.extend(tool_messages)


    final = hf_client.chat.completions.create(
        model="openai/gpt-oss-120b:novita",
        messages=messages
    )
    

    print("Final response:", final.choices[0].message.content)
    


if __name__ == "__main__":
    asyncio.run(main())