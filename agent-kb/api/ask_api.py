import time
import uuid
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel
from agent.agent import build_agent

router = APIRouter()
_agent_executor = None


class AskRequest(BaseModel):
    question: str


def get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor


@router.post("/ask")
def ask_question(request: AskRequest):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    print("\n" + "=" * 70)
    print(f"[ASK] START  request_id={request_id}")
    print(f"[ASK] start_time={datetime.now().isoformat()}")
    print(f"[ASK] QUESTION: \"{request.question}\"")
    print("=" * 70)

    executor = get_agent_executor()
    result = executor.invoke({
        "messages": [{"role": "user", "content": request.question}]
    })

    messages = result["messages"]
    answer = messages[-1].content

    print(f"\n[ASK] agent message trace ({len(messages)} messages):")
    for i, m in enumerate(messages):
        mtype = getattr(m, "type", type(m).__name__)
        calls = getattr(m, "tool_calls", None)
        if calls:
            for c in calls:
                print(f"[ASK]   [{i}] {mtype} -> TOOL CALL "
                      f"{c.get('name')} args={c.get('args')}")
        else:
            preview = str(m.content)[:80].replace("\n", " ")
            print(f"[ASK]   [{i}] {mtype}: \"{preview}...\"")

    print("\n" + "=" * 70)
    print(f"[ASK] END    request_id={request_id}")
    print(f"[ASK] end_time={datetime.now().isoformat()}")
    print(f"[ASK] ANSWER: \"{answer}\"")
    print(f"[ASK] answer_length={len(answer)} chars")
    print(f"[ASK] TOTAL={time.time() - start:.2f}s")
    print("=" * 70 + "\n")

    return {"question": request.question, "answer": answer}