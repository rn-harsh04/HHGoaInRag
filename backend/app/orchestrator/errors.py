"""Stage-level pipeline errors."""

from enum import Enum


class StageName(str, Enum):
    VALIDATE_AUDIO = "validate_audio"
    STT = "stt"
    PRE_GUARDRAIL = "pre_guardrail"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    POST_GUARDRAIL = "post_guardrail"


class PipelineError(Exception):
    def __init__(
        self,
        stage: StageName,
        message: str,
        *,
        http_status: int = 500,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.http_status = http_status
        self.user_message = user_message or message
