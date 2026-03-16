from pydantic import ValidationError

from switchyard.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    FinishReason,
    UsageStats,
)


def test_chat_request_and_response_serialize() -> None:
    request = ChatCompletionRequest(
        model="mock-chat",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
        max_output_tokens=128,
    )
    response = ChatCompletionResponse(
        id="resp_123",
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role=ChatRole.ASSISTANT, content="Hi there"),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage=UsageStats(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        backend_name="mock-backend",
    )

    payload = response.model_dump(mode="json")

    assert request.messages[0].role is ChatRole.USER
    assert payload["backend_name"] == "mock-backend"
    assert payload["choices"][0]["message"]["content"] == "Hi there"


def test_chat_response_rejects_non_assistant_choice() -> None:
    try:
        ChatCompletionChoice(
            index=0,
            message=ChatMessage(role=ChatRole.USER, content="nope"),
            finish_reason=FinishReason.STOP,
        )
    except ValidationError as exc:
        assert "assistant" in str(exc)
    else:
        raise AssertionError("ChatCompletionChoice should reject non-assistant messages")
