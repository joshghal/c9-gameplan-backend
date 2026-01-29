"""LLM client supporting multiple providers (Anthropic, ASI:One)."""

from typing import AsyncGenerator, Optional
from functools import lru_cache
import httpx

from ...config import get_settings


class LLMClient:
    """Unified async LLM client supporting multiple providers."""

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider
        self.model = self.settings.llm_model

        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        else:
            # ASI:One or other OpenAI-compatible providers
            self.api_key = self.settings.asi1_api_key
            self.api_base = self.settings.asi1_api_base
            self.client = httpx.AsyncClient(timeout=60.0)

    async def chat(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[list] = None,
    ):
        """Send a chat message and get a response."""
        if self.provider == "anthropic":
            return await self._anthropic_chat(messages, system, max_tokens, temperature, tools)
        else:
            return await self._openai_compatible_chat(messages, system, max_tokens, temperature, tools)

    async def _anthropic_chat(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[list],
    ):
        """Anthropic Claude API call."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)
        return response

    async def _openai_compatible_chat(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[list],
    ):
        """OpenAI-compatible API call (ASI:One, etc.)."""
        # Prepend system message
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            # Convert Anthropic tool format to OpenAI format
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    }
                })
            payload["tools"] = openai_tools

        response = await self.client.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        return OpenAIResponse(response.json())

    async def stream_chat(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: Optional[list] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response for real-time display."""
        if self.provider == "anthropic":
            async for chunk in self._anthropic_stream(messages, system, max_tokens, temperature, tools):
                yield chunk
        else:
            async for chunk in self._openai_compatible_stream(messages, system, max_tokens, temperature, tools):
                yield chunk

    async def _anthropic_stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[list],
    ) -> AsyncGenerator[str, None]:
        """Stream from Anthropic."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _openai_compatible_stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
        temperature: float,
        tools: Optional[list],
    ) -> AsyncGenerator[str, None]:
        """Stream from OpenAI-compatible API."""
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with self.client.stream(
            "POST",
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except:
                        continue

    async def chat_with_tools(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict]]:
        """Chat with function calling support."""
        response = await self.chat(
            messages=messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
        )

        if self.provider == "anthropic":
            text_content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text_content = block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            return text_content, tool_calls
        else:
            # OpenAI format
            choice = response.data.get("choices", [{}])[0]
            message = choice.get("message", {})
            text_content = message.get("content", "") or ""
            tool_calls = []

            for tc in message.get("tool_calls", []):
                import json
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                })

            return text_content, tool_calls

    async def close(self):
        """Close the client."""
        if self.provider != "anthropic":
            await self.client.aclose()


class OpenAIResponse:
    """Wrapper to make OpenAI response look like Anthropic response."""

    def __init__(self, data: dict):
        self.data = data
        self.content = [OpenAIContentBlock(data)]


class OpenAIContentBlock:
    """Wrapper for content block."""

    def __init__(self, data: dict):
        self.type = "text"
        choice = data.get("choices", [{}])[0]
        self.text = choice.get("message", {}).get("content", "") or ""


# Backwards compatibility
AnthropicClient = LLMClient


@lru_cache()
def get_anthropic_client() -> LLMClient:
    """Get cached LLM client instance."""
    return LLMClient()
