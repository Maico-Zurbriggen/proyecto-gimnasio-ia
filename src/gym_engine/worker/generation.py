from __future__ import annotations

from vercel.queue import Message, Topic, subscribe

from ..config import Settings
from ..contracts import GenerationMessage
from ..service import process_generation

generation_topic = Topic[GenerationMessage]("gym-routine-generations")


@subscribe(
    topic=generation_topic,
    consumer_group="routine-generation-worker-v1",
    retry_after=150,
    max_concurrency=1,
    max_attempts=6,
)
async def generate_routine(message: Message[GenerationMessage]) -> None:
    await process_generation(message.payload.request_id, Settings.from_env())
