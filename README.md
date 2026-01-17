# Obsidian RAG API

API и MCP серверы для интеллектуальной работы с Obsidian vault'ами через RAG (Retrieval-Augmented Generation).

## Что это?

Этот подмодуль предоставляет три основных компонента:

1. **FastAPI REST API** — HTTP endpoints для загрузки vault'ов и запросов к агенту
2. **MCP Server (stdio)** — для интеграции с Claude Desktop и другими MCP клиентами
3. **MCP Server (HTTP)** — HTTP версия MCP сервера для кастомных интеграций
4. **LangGraph Agent** — интеллектуальный агент с рекурсивным расширением контекста

## Компоненты

### 📁 Структура

```
obsidian-rag-api/
├── src/obsidian_rag_api/
│   ├── __init__.py
│   ├── domain/            # Доменные сущности (SQLAlchemy модели)
│   ├── application/       # Сценарии и сервисы (auth, vaults, agent)
│   ├── infrastructure/    # DB/LLM конфиги и адаптеры
│   ├── interfaces/        # HTTP/MCP интерфейсы
│   ├── api.py             # Wrapper: FastAPI REST endpoints
│   ├── server.py          # Wrapper: MCP Server (stdio transport)
│   └── server_http.py     # Wrapper: MCP Server (HTTP transport)
└── pyproject.toml
```

### 🔧 config.py

Централизованная конфигурация через Pydantic Settings:

- `OPENAI_API_KEY` — ключ для OpenAI API
- `OPENAI_BASE_URL` — опциональный base URL (для совместимых API)
- `OPENAI_MODEL` — основная модель (по умолчанию: gpt-4o-mini)
- `OPENAI_CHEAP_MODEL` — модель для проверок релевантности
- `MAX_RECURSION_DEPTH` — макс. глубина расширения контекста (по умолчанию: 3)
- `SEARCH_TOP_K` — количество результатов поиска (по умолчанию: 5)
- `EMBEDDINGS_MODEL` — модель для embeddings (по умолчанию: openai/text-embedding-3-large)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — подключение к Neo4j
- `QDRANT_HOST`, `QDRANT_PORT` — подключение к Qdrant

### 🗂️ vault_manager.py

Управление множественными Obsidian vault'ами:

- **UUID-based isolation** — каждый vault получает уникальный ID
- **In-memory storage** — vault'ы хранятся в памяти (теряются при перезапуске)
- **obsidian_retriever integration** — создает KnowledgeBaseManager для каждого vault

**Основные методы:** `upload_vault`, `upload_vault_with_progress`, `get_or_create_vault_manager`

### 🤖 agent.py

LangGraph агент с умным RAG и рекурсивным расширением контекста.

#### Граф состояний

```
START → reformulate → search → check_context ⟷ extend_context → generate_answer → END
```

#### State

```python
class AgentState(TypedDict):
    original_query: str           # Исходный запрос
    vault_id: str                 # UUID vault'а
    reformulated_query: str       # Переформулированный запрос
    search_results: list[dict]    # Результаты поиска
    knowledge_base: list[Note]    # Накопленные заметки
    explored_notes: set[str]      # Уже изученные заметки
    current_depth: int            # Текущая глубина рекурсии
    final_answer: str             # Итоговый ответ
```

#### Узлы графа

1. **reformulate** — переформулирует запрос для лучшего поиска
2. **search** — семантический поиск в Qdrant
3. **check_context** — проверяет достаточность информации для ответа
4. **extend_context** — параллельно проверяет связанные заметки через LLM
5. **generate_answer** — генерирует финальный ответ

#### Функции

```python
# Создать и скомпилировать граф
graph = create_agent_graph()

# Запустить агент
result = await graph.ainvoke({
    "original_query": "Что такое квантовая механика?",
    "vault_id": "550e8400-e29b-41d4-a716-446655440000"
})
```

### 🌐 api.py

FastAPI REST API с endpoint'ами для работы с vault'ами и агентом.

#### Endpoints

##### POST /upload

Загружает ZIP-архив с Obsidian vault.

**Request (multipart/form-data):**
- `file` — ZIP файл
- `include_paths` (optional) — пути для включения (через запятую)
- `exclude_paths` (optional) — пути для исключения
- `chunk_size` (optional) — размер чанков (по умолчанию: 500)

**Response:**
```json
{
  "vault_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Vault uploaded successfully",
  "status": "success"
}
```

##### POST /upload/stream

То же самое, но с прогрессом через Server-Sent Events.

**SSE события:**
```json
{"stage": "extracting", "progress": 15, "message": "...", "elapsed": "2s", "eta": "11s"}
{"stage": "parsing", "progress": 50, "message": "..."}
{"stage": "processing", "progress": 65, "message": "..."}
{"stage": "storing", "progress": 82, "message": "..."}
{"stage": "linking", "progress": 100, "message": "..."}
{"stage": "complete", "vault_id": "...", "notes_count": 42}
```

##### GET /vaults

Возвращает список всех vault ID.

**Response:**
```json
{
  "vaults": ["550e8400-...", "660e9511-..."],
  "count": 2
}
```

##### POST /agent

Запрос к агенту.

**Request:**
```json
{
  "query": "Расскажи про машинное обучение",
  "vault_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "optional-session-id"
}
```

**Response:**
```json
{
  "query": "Расскажи про машинное обучение",
  "reformulated_query": "machine learning algorithms neural networks",
  "answer": "На основе ваших заметок...",
  "notes_used": 5,
  "max_depth_reached": 2,
  "thread_id": "abc123",
  "vault_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

##### POST /agent/stream

Стриминг прогресса агента через SSE.

**Request:** тот же, что и `/agent`

**SSE события:**
```json
{"stage": "reformulating", "message": "Reformulating query..."}
{"stage": "searching", "message": "Found 5 relevant notes"}
{"stage": "extending", "message": "Checking 3 related notes..."}
{"stage": "generating", "message": "Generating final answer..."}
{"stage": "complete", "answer": "...", "notes_used": 8}
```

### 🔌 server.py

MCP Server с stdio транспортом для интеграции с Claude Desktop.

#### Tools

| Tool | Описание | Параметры |
|------|----------|-----------|
| `list_vaults` | Возвращает список доступных vault'ов | `query` (опц.), `limit` (опц.) |
| `read_note` | Читает заметку по ID | `vault_id`, `note_id` |
| `search` | Семантический поиск | `vault_id`, `query`, `top_k` |
| `extend_context_using_nearest` | Расширяет контекст | `vault_id`, `note_id`, `query` |

#### Запуск

```bash
python -m obsidian_rag_api.server
```

Или через главный entry point:
```bash
python main.py mcp
```

### 🌍 server_http.py

MCP Server с HTTP транспортом (SSE) на порту 8001.

**Авторизация MCP:** для HTTP используйте `Authorization: Bearer <MCP_TOKEN>`, для stdio задайте `MCP_AUTH_TOKEN` в окружении. Токен можно получить в личном кабинете веб‑интерфейса.

#### Endpoints

- `GET /sse` — SSE endpoint для MCP протокола
- `POST /messages` — отправка сообщений серверу

#### Запуск

```bash
python -m obsidian_rag_api.server_http
```

Или:
```bash
python main.py mcp --http
```

### 🧠 llm.py

Утилиты для работы с LLM:

```python
from obsidian_rag_api.llm import get_chat_model, get_cheap_chat_model

# Основная модель
llm = get_chat_model()

# Дешевая модель для проверок
cheap_llm = get_cheap_chat_model()

# Использование
response = llm.invoke("Your prompt here")
```

## Использование как библиотека

### Установка

```bash
# Как часть workspace
cd /path/to/obsidian-rag-mcp
uv sync

# Standalone (если опубликован в PyPI)
uv add obsidian-rag-api
```

### Пример

```python
from obsidian_rag_api.application.vault_manager import create_vault_manager
from obsidian_rag_api.application.agent import run_agent_with_vault

vault_id = "your-vault-id"
manager = create_vault_manager(vault_id)
manager.init_vault_from_zip("/path/to/vault.zip")

result = await run_agent_with_vault("Что такое квантовая механика?", vault_id)
print(result["answer"])
```

## Зависимости

```toml
fastapi>=0.124.4
langchain>=1.1.3
langchain-openai>=1.1.3
langgraph>=1.0.5
mcp>=1.24.0
neomodel>=6.0.0
qdrant-client>=1.16.2
sentence-transformers>=5.2.0
uvicorn>=0.38.0
pydantic>=2.12.5
python-dotenv>=1.2.1
```

## Переменные окружения

Создайте `.env` файл:

```env
# OpenAI
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_CHEAP_MODEL=gpt-4o-mini

# Agent
MAX_RECURSION_DEPTH=3
SEARCH_TOP_K=5

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embeddings
EMBEDDINGS_MODEL=openai/text-embedding-3-large
```

## Запуск серверов

### Все вместе (Agent API + MCP HTTP)

```bash
python main.py all
```

Это запустит:
- Agent API на порту 8000
- MCP HTTP на порту 8001

### Только Agent API

```bash
python main.py agent
```

### Только MCP (stdio)

```bash
python main.py mcp
```

### Только MCP (HTTP)

```bash
python main.py mcp --http
```

## Интеграция с Claude Desktop

Добавьте в `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "docker",
      "args": [
        "compose", "exec", "-T", "obsidian-rag",
        "uv", "run", "python", "main.py", "mcp"
      ]
    }
  }
}
```

## Development

### Запуск тестов

```bash
uv run pytest
```

### Форматирование

```bash
uv run ruff format .
uv run ruff check --fix .
```

## Лицензия

MIT
