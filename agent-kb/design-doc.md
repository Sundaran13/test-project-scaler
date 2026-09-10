Knowledge Base Agent — Design
A document Q&A agent built from open-source parts. Ingest documents through an API, ask questions through another, get answers grounded in the ingested content with source citations — and an honest refusal when the answer isn't there.
Everything runs locally. The only external call is the LLM (Groq free tier).

1. What it does
Two API surfaces over one knowledge base:
Ingestion — accepts a PDF, text file, or pasted text; processes it asynchronously; returns a task id to poll.
Query — accepts a question; an agent decides whether to search the knowledge base, then answers from what it retrieves.
Eval harness — a fixed question set that scores the agent, used to measure whether a change helped or hurt.
—-------------Below mcp is implemented but i need to still test it—-----
MCP server — exposes the same retrieval over the Model Context Protocol, so external clients (Claude Desktop) can use the knowledge base directly.

2. Architecture
                   ┌─────────────────────┐
                    │   Ingestion API      │
                    │  POST /ingest/*      │
                    │  returns task_id     │
                    └──────────┬───────────┘
                               │ queue
                    ┌──────────▼───────────┐
                    │       Redis          │
                    └──────────┬───────────┘
                               │ poll
                    ┌──────────▼───────────┐
                    │   Celery worker      │
                    │  loader → chunker    │
                    │  → embedder          │
                    └──────────┬───────────┘
                               │ write
                    ┌──────────▼───────────┐
                    │  Chroma (embedded)   │
                    │  SQLite + HNSW       │
                    └─────┬──────────┬─────┘
                     read │          │ read
              ┌───────────▼──┐   ┌───▼──────────┐
              │  Query API   │   │  MCP server  │
              │  POST /ask   │   │   (stdio)    │
              │  LangChain   │   │              │
              │  agent+Groq  │   │  any client  │
              └──────────────┘   └──────────────┘

Processes
Five, none sharing memory:
Process
Started by
Chroma access
uvicorn
manual
read
Celery worker
manual
write
Redis
brew service
none
MCP server
Claude Desktop
read
Eval script
manual
read

Four processes open the same embedded database from four different working directories. This drove several design decisions — see §7.

3. File layout
agent-kb/
├── main.py                   FastAPI app, mounts routers, loads .env
├── celery_app.py             Celery config, include=["tasks"]
├── tasks.py                  background task definitions
├── mcp_server.py             MCP entry point (independent of FastAPI)
├── .env                      GROQ_API_KEY
│
├── ingestion/
│   ├── loader.py             file/text → LangChain Document
│   ├── chunker.py            Document → overlapping chunks
│   └── embedder.py           MiniLM singleton
│
├── knowledge_base/
│   └── vector_store.py       Chroma read/write, relevance threshold
│
├── agent/
│   ├── tools.py              search() wrapped as a LangChain tool
│   └── agent.py              Groq + tool + system prompt
│
├── api/
│   ├── ingest_api.py         POST /ingest/{text,file}, GET /ingest/status
│   └── ask_api.py            POST /ask
│
├── evals/
│   ├── dataset.json          questions + expected keywords + answerable flag
│   ├── run_eval.py           runner, scorer, report
│   └── results.json          output, for comparing runs
│
├── chroma_db/                the knowledge base on disk
└── uploads/                  temp staging, deleted after ingestion

Responsibilities
File
Does
loader.py
Normalizes PDF, txt, and raw text into Document objects. Only place that knows about file formats.
chunker.py
Splits Documents into ~500-char overlapping chunks at semantic boundaries.
embedder.py
Holds the MiniLM model as a singleton (90MB, loaded once).
vector_store.py
Owns PERSIST_DIR, add_documents(), search(), relevance filtering.
tools.py
@tool-wrapped search(). Docstring is what the LLM reads to decide when to call it.
agent.py
Binds Groq + the tool + system prompt into an agent.
ingest_api.py
Saves uploads, queues tasks, returns ids, polls status.
ask_api.py
Runs the agent, extracts the final message.
tasks.py
The background pipeline. Load → chunk → embed → store → cleanup.


4. Data model
Every chunk is one Chroma record with four fields:
id         "6523efd3-141d-46e9-a108-4c8b8617f6fa"
embedding  [0.21, -0.44, 0.87, ...]        384 floats
document   "Must-haves: Built and shipped AI products..."
metadata   {"source": "...", "page": 0, "title": "scaler_kb", ...}

The split matters:
Field
Role
embedding
Searched. kNN runs on this. Never shown.
document
Returned. Goes into the prompt. Never searched directly.
metadata
Cited. Structured fields, exact match.
id
Addressed. Update or delete a specific chunk.

You search vectors and return text. Keeping these separate is the core idea.
Metadata origin
Set once in the loader, then copied to every chunk by the splitter. PyPDFLoader supplies source, page, page_label, total_pages, title. Raw text supplies just source (the caller's source_name).
That copy is what makes per-chunk citation possible.
On disk
chroma_db/
├── chroma.sqlite3      ids, text, metadata — plain SQLite
└── <uuid>/
    ├── data_level0.bin  vectors
    └── link_lists.bin   HNSW graph

HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbor index. Exact search would be one distance calculation per stored vector; HNSW visits a small fraction of the graph instead. The approximation is almost always the right trade.

5. Write path
POST /ingest/file
   ↓ uvicorn: save to uploads/<uuid>.pdf, .delay() the task, RETURN task_id
   ↓ Redis: hold the message
   ↓ worker: pick up (threads pool)
   ↓ loader: PDF → one Document per page
   ↓ chunker: pages → N overlapping chunks, metadata copied
   ↓ embedder: each chunk → 384 numbers (MiniLM, local)
   ↓ Chroma: SQLite + HNSW
   ↓ delete temp file
   ↓ result → Redis, retrievable by task_id

The HTTP response returns after step 2, in milliseconds. Embedding a large PDF takes seconds; doing it inside the request would block a worker and risk a browser timeout.
The file is written to disk before queuing because Celery serializes task arguments to JSON through Redis — a path string survives that, a file object doesn't.
Chunking strategy
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

Recursive means the separator list is a preference order. It tries paragraph breaks first; for any piece still over 500 chars it tries line breaks, then sentence ends, then word boundaries, then raw characters as a guaranteed exit. Cuts land at the highest-meaning boundary available.
Overlap exists because a clean cut can bisect a sentence, leaving neither chunk able to answer a question about it. Each chunk repeats the previous chunk's tail.
500 is bounded by the embedding model: MiniLM truncates around 256 tokens (~1000 chars), and anything past that is silently discarded before embedding. 500 stays comfortably inside. It has not been tuned empirically — see §8.

6. Read path
POST /ask
   ↓ LangChain sends Groq: system prompt + question + tool schema
   ↓ Groq responds with a TOOL CALL, not an answer
   ↓ tool runs locally: query → MiniLM → HNSW → top-k → threshold filter
   ↓ chunks return to the conversation as a tool result
   ↓ Groq writes a grounded answer with citations
   ↓ take messages[-1].content

Two LLM round trips, one embedding, one vector search.
Agent, not chain
A fixed RAG chain retrieves unconditionally on every query. Here the LLM gets retrieval as a tool and decides whether and how often to use it. Observed in practice: asked "who is srk?", the agent searched "Shah Rukh Khan", got nothing useful, then searched "Shah Rukh Khan biography" before refusing. A chain would have searched once.
The tradeoff is real — an extra round trip, less predictable, harder to test. For a single-corpus Q&A system a fixed chain would have sufficed. The agent was chosen to understand the decision-making layer.
Grounding
The system prompt does three jobs, each fixing a specific failure:
Always use the tool — otherwise the model answers from training data
Say so honestly when nothing relevant is found — anti-hallucination
Cite sources inline — makes claims auditable
The tool's docstring is the other lever. It is serialized into the prompt and is what the model reads to decide when the tool applies — prompt engineering in a comment's clothing.
Relevance threshold
RELEVANCE_THRESHOLD = None   # calibration in progress

similarity_search_with_score returns squared L2 distance — lower is closer. For MiniLM's normalized vectors the practical range is 0 to 2, where distance = 2 - 2 × cosine_similarity.
Measured so far: irrelevant queries score 1.55–1.76 (cosine ~0.12–0.22, essentially noise). Distances for relevant queries have not yet been measured, so no threshold is set. An earlier attempt at 1.0 correctly rejected noise but also filtered out genuine founder-bio chunks — the failure looked identical to "the knowledge base doesn't have it", which is why calibration needs both bands, not just the reject case.

7. Decisions worth recording
Absolute paths derived from __file__
PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db"
)

"./chroma_db" resolves against the process working directory, not the file. uvicorn starts in the project folder so it worked; Claude Desktop launches the MCP server from elsewhere and Chroma tried to create its database at the filesystem root — Read-only file system (os error 30).
__file__ never changes. Four processes, four possible start directories, one database. Same fix applied to UPLOAD_DIR, since the API writes the file and a different process reads it.
Celery threads pool, not prefork
celery -A celery_app worker --pool=threads --concurrency=4

The default prefork pool calls fork() for real parallelism, since the GIL prevents threads from running Python concurrently. But fork() copies all memory while only the calling thread survives. The ingestion task loads PyTorch, which initializes Apple's Metal framework across multiple threads; forking mid-initialization left the child with a half-built GPU object and none of the threads that were building it. macOS aborts rather than continue:
+[MPSGraphObject initialize] may have been in progress in another thread
when fork() was called. Crashing instead.

Threads avoid the fork entirely. The GIL concern mostly doesn't apply here — the expensive work is inside PyTorch's C++ and Chroma's Rust, both of which release the GIL.
include=["tasks"] in the Celery constructor
Without it the worker booted with an empty [tasks] registry. The API could send tasks (it imports tasks.py directly, so it knows the names) but the worker had no function to run — messages arrived and were discarded with KeyError: 'ingest_text_task'.
Singletons for the embedder and vector store
Loading 90MB of model weights takes seconds. Without the singleton it would reload per chunk, turning a 1-second ingestion into minutes.
MCP server imports nothing from FastAPI or the agent
from mcp.server.mcpserver import MCPServer
from knowledge_base.vector_store import search

Two imports. The point isn't difficulty — it's that the knowledge base becomes a service consumable by clients that know nothing about this codebase, rather than a feature locked inside one app.

8. Evaluation
dataset.json → run_eval.py → agent → score → report + results.json

Twelve questions: roughly two-thirds answerable, one-third deliberately not.
The two groups are scored in opposite directions:
if case["answerable"]:
    passed = score_keywords(answer, case["must_contain"])   # must answer
else:
    passed = looks_like_refusal(answer)                     # must refuse

For unanswerable questions, producing an answer is the failure. An all-answerable eval set structurally cannot detect hallucination — a system that confidently invents answers scores 100% on one.
Retrieval is measured separately from generation:
def retrieval_hit(question, must_contain, k=4):
    docs = search(question, k=k)     # bypasses the LLM entirely

A failure then arrives with a diagnosis. Retrieval miss means the model never had a chance — fix chunking, k, or the embedding model. Retrieved but wrong answer means retrieval is fine — fix the prompt or the model. Without the split you're guessing which half to work on.
Known weakness: refusal detection matches against a phrase list, so an unanticipated phrasing scores as a failure even when the agent behaved correctly. That's a false negative in the evaluator, not the system. The fix is LLM-as-judge for semantic matching, which handles paraphrase but costs money and carries its own biases (judges reward longer, more confident answers). The keyword version was built first deliberately; the useful signal is where the two disagree.

9. Known gaps
Gap
Impact
Fix
No eval numbers
Chunk size, k, and overlap are untuned defaults. No baseline to compare against.
Run the harness across three chunk sizes, record results.
Threshold uncalibrated
Currently None. Only the noise band has been measured.
Measure relevant-query distances, set the cutoff in the gap.
PDF text is mangled
Extraction produces "them in unlocking" — multiple spaces between words. Breaks the ". " separator, degrades embeddings.
Regex cleanup in the loader, then re-ingest.
No metadata filtering
Metadata is used for citations only. No scoping, no tenant isolation.
Pass a filter to similarity_search. Required before multi-tenancy.
No deduplication
Re-ingesting a document duplicates every chunk. Duplicates waste retrieval slots.
Deterministic ids derived from content or source.
No update/delete path
Changed documents leave stale chunks competing in retrieval forever.
Delete by source metadata before re-ingesting.
Chroma is single-process
uvicorn, worker, MCP server, and eval script contend for one SQLite-backed store.
Chroma client-server mode, or Postgres + pgvector.
Citations point at temp filenames
The source metadata is the deleted upload path. The title field is a better fallback but still not a resolvable link.
Store a stable identifier (URL or document id) at ingestion time.
Print-based logging
No levels, no file output, per-chunk prints would bury a large ingestion.
Structured JSON logging with request ids and chunk ids.
No error handling on /ask
Failures return a bare 500.
try/except with meaningful messages.
No conversation memory
Every /ask is stateless; follow-ups don't work.
session_id + LangGraph checkpointing.
No groundedness check
The prompt asks for grounding; nothing verifies it. The model could cite correct chunks and still assert something they don't support.
Verification pass — check each claim is entailed by retrieved context.


10. Operating notes
Running it
# 1. Redis
brew services start redis

# 2. Celery worker — threads pool is required
cd agent-kb && source .venv/bin/activate
export HF_HUB_OFFLINE=1
celery -A celery_app worker --loglevel=info --pool=threads --concurrency=4

# 3. API
cd agent-kb && source .venv/bin/activate
uvicorn main:app

--reload watches .venv by default and will restart the server hundreds of times. Use --reload-exclude or omit it.
Where the logs are
Action
Terminal
Output
POST /ingest/*
Celery worker
[INGEST], [LOADER], [CHUNKER], [STORE]
POST /ingest/*
uvicorn
one access-log line
POST /ask
uvicorn
[ASK], [SEARCH], message trace
MCP
~/Library/Logs/Claude/mcp-server-knowledge-base.log
same prints, redirected

MCP client config
~/Library/Application Support/Claude/claude_desktop_config.json:
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/abs/path/agent-kb/.venv/bin/python",
      "args": ["/abs/path/agent-kb/mcp_server.py"],
      "cwd": "/abs/path/agent-kb",
      "env": {
        "HF_HOME": "/abs/path/agent-kb/.hf_cache",
        "HF_HUB_OFFLINE": "1"
      }
    }
  }
}

Stop uvicorn and the Celery worker before using MCP — Chroma is single-process. Claude Desktop only restarts the server when Claude Desktop itself restarts, so code changes need a full Cmd+Q and reopen.

11. The pattern behind the bugs
Bug
Assumed
Reality
./chroma_db
started from the project dir
Claude Desktop started elsewhere
Celery SIGABRT
single process
Celery forked after Metal init
Empty [tasks]
both sides share code
worker never imported tasks.py
KeyError: GROQ_API_KEY
env vars persist
new terminal, gone
pypdf not found
requirements complete
venv rebuild dropped it

None were logic errors. Every function was correct in isolation. What failed were assumptions about the environment — each breaking the moment a new process, terminal, or virtualenv appeared.
That distinction is most of what separates code that runs on one machine from code that runs anywhere.

