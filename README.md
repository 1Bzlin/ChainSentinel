# ChainSentinel

面向供应链风险管理与采购决策的工程化 **Agent Harness 平台**。ChainSentinel 在 RBAC 权限约束下融合 Secure RAG、GraphRAG、Memory 与 MCP Tools，并通过 LangGraph 将请求拆解为可验证、可追踪的执行计划，为采购、物流、质量、财务与管理层提供 **可审计、可引用、可评估** 的风险洞察与决策支持。

当前版本的重点是 **Planner V1**：简单请求走确定性规则计划，复杂请求由 LLM 生成最多 8 个任务的 DAG；计划在执行前必须通过结构、实体、租户范围、角色权限和高风险审批校验。

## 系统架构


### 架构分层说明

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| **数据来源** | 企业文档、结构化业务数据、外部开源数据集、用户上传；经 ETL 解析、标准化、分块 | `packages/data_adapters/` |
| **Secure RAG** | 文档智能检索；在 Qdrant 层做 tenant + 密级 + 角色/供应商 scope 过滤 | Qdrant、Embedding API |
| **GraphRAG** | 供应链知识图谱多跳推理；预置 Cypher 模板（非 Text2Cypher） | Neo4j |
| **Agent 编排** | 所有请求生成可审计执行计划；支持 RAG、Graph、Memory、Tool 多任务依赖与并行执行 | LangGraph、DeepSeek |
| **Memory** | Redis 短期会话记忆 + PostgreSQL 长期记忆；支持多轮追问 | Redis、PostgreSQL |
| **MCP Tools** | 供应商/物流/报告导出等结构化工具；角色白名单 + 高风险审批 | `packages/mcp_servers/` |
| **可视化输出** | Next.js 控制台：Dashboard、RAG/GraphRAG 工作台、Agent Trace、Benchmark | Next.js 15、ECharts、React Flow |
| **基础设施** | JWT + RBAC、Audit Log、trace_id 全链路可观测、确定性 Benchmark | FastAPI、Docker Compose |

### 与普通 RAG Demo 的区别

- **权限在检索层**：Filter 在 Qdrant / Neo4j / Tool 入口执行，而非 Prompt 自律
- **受约束规划**：规则快速生成简单计划，LLM 拆解复杂任务，随后逐任务做结构、实体与权限校验
- **多源证据**：文档向量 + 图谱路径 + 记忆 + 结构化 Tool，统一聚合后生成报告
- **可观测与评测**：每次请求绑定 `trace_id`；111 条基础评测 + 150 条 Agent Plan V2 评测覆盖规划、编排、权限、Memory、回答忠实性与审计

---

## 核心流程

### 1. Chat 主路径（`POST /api/v1/chat`）

LangGraph Agent 工作流的完整链路：

```
Permission → Context Loader → Planner → Plan Validator
  ├─ blocked ───────────────────────────────→ Review
  ├─ clarification_required → Clarify ─────→ Review
  └─ ready → Plan Executor[RAG | Graph | Tool | Memory]*
  → Aggregator → Report → Review → Memory Writer
```

| 节点 | 作用 |
|------|------|
| `permission` | 查询级权限检查；拒绝则跳转 review |
| `context_loader` | 加载租户/用户/会话隔离的短期上下文并补全指代 |
| `planner` | 所有请求生成统一计划；复杂请求拆成多个可审计任务 |
| `plan_validator` | 校验任务 DAG、实体、租户范围、每个任务的权限及高风险审批要求 |
| `plan_executor` | 按依赖分批执行 RAG、Graph、Tool、Memory；同批任务并行 |
| `clarify` | 实体或意图无法唯一确定时返回澄清问题，不执行不安全或不完整的计划 |
| `aggregator` | 多源证据合并、去重、上下文预算裁剪 |
| `report` | LLM 生成结构化报告 + citations |
| `review` | 规则审查（引用存在性、blocked 处理） |
| `memory_writer` | 写入会话/长期记忆 |

简单问题形成单任务规则计划；复杂问题形成多任务计划，不再使用 `hybrid` 等组合路由枚举。每次运行都会持久化 `AgentExecutionPlan` 与 `AgentTaskRun`，并在 Trace 页面展示计划、任务状态、检索记录、工具调用和节点耗时。

### 2. Secure RAG 路径（`POST /api/v1/rag/query`）

```
JWT 认证 → build_qdrant_permission_filter(user_context)
  → embed(query) → Qdrant Top-K（Cosine，MVP 无 Rerank）
  → PostgreSQL 加载 chunk 正文 → LLM 生成 + citations
```

Qdrant payload 含 `tenant_id`、`security_level_rank`、`access_scope`、`allowed_roles`、`supplier_id` 等，**检索前过滤**，非检索后裁剪。

### 3. GraphRAG 路径（`POST /api/v1/graphrag/query`）

```
JWT → 角色 query_type 权限检查 → 选择 Cypher 模板
  → Neo4j 多跳查询 → 路径解析 → 返回影响链
```

**图谱节点：** Supplier · Material · Product · Order · Customer · RiskEvent · DeliveryEvent · QualityIssue

**典型路径：** `Supplier → Material → Product → Order → Customer`

**Cypher 模板：** supplier_impact · product_dependency · supplier_alternatives · delivery_delay_impact · quality_issue_trace · customer_order_dependency 等

### 4. 文档入库（`POST /api/v1/documents/upload`）

```
上传 txt/md/pdf → data/raw/ → 解析分块 → Embedding
  → Qdrant upsert + PostgreSQL documents/chunks 双写（含权限 metadata）
  → 解析文本落盘 data/processed/
```

---

## 技术栈

**后端：** Python 3.11 · FastAPI · LangGraph · SQLAlchemy · Alembic  
**前端：** Next.js 15 · React 19 · TypeScript · Tailwind · Zustand · React Query  
**存储：** PostgreSQL · Qdrant · Neo4j · Redis · 本地 `data/`（MinIO 预留）  
**AI：** DeepSeek / LiteLLM / vLLM（LLM）· SiliconFlow 等（Embedding）  
**部署：** Docker Compose · uv 包管理

---

## 环境要求

- Python >= 3.11，[uv](https://github.com/astral-sh/uv)
- Node.js 18+（前端）
- Docker、Docker Compose 与 [crane](https://github.com/google/go-containerregistry/tree/main/cmd/crane)（Postgres / Redis / Qdrant / Neo4j / MinIO）
- 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`、`EMBEDDING_*`、`JWT_SECRET_KEY`、`EMBEDDING_DIM`

---

## 快速开始

```bash
git clone https://github.com/1Bzlin/ChainSentinel.git
cd ChainSentinel

uv sync
cp .env.example .env          # 填入 API Key

make up                       # 复用本地镜像；缺失镜像通过 crane 拉取
make migrate
make seed
make ingest-seed-docs
make build-graph

make api                      # 终端 1：FastAPI :8000
make web-install && make web  # 终端 2：Next.js :3000
```

**冒烟测试：**

```bash
make smoke-mvp3               # Secure RAG
make smoke-graphrag           # GraphRAG
make smoke-benchmark          # Benchmark
uv run pytest tests/test_planning.py
```

**Demo 账号：** `admin@example.com` / `password123`（仅开发环境）

---

## API 示例

### 登录

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"password123"}'
```

### Secure RAG

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"Alpha Components 合同付款条款是什么？","top_k":5}'
```

### GraphRAG

```bash
curl -X POST http://localhost:8000/api/v1/graphrag/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "供应商 SUP-001 延迟会影响哪些订单和客户？",
    "query_type": "supplier_impact",
    "entity_code": "SUP-001",
    "top_k": 20
  }'
```

### Chat（LangGraph 完整工作流）

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"请总结 Alpha Components 合同中的付款条款。"}'
```

返回除 `answer`、`citations`、`trace_id`、`nodes_executed` 外，还包含 `plan_id`、`plan_status`、`planning_source`、`planning_confidence`、`plan_summary` 与 `resolved_query`，可直接审计规划和任务执行结果。

---

## 项目结构

```
ChainSentinel/
├── apps/ 
│   ├── api/              # FastAPI 入口、路由、中间件
│   └── web/              # Next.js 前端控制台
├── packages/
│   ├── agents/           # LangGraph、Planner、Validator、并行任务执行与证据聚合
│   ├── rag/              # 文档摄取、Secure RAG
│   ├── graphrag/         # Neo4j 模板 Cypher、建图
│   ├── memory/           # Redis 短期 + PG 长期记忆
│   ├── authz/            # RBAC、Qdrant filter、Tool policy
│   ├── mcp_servers/      # MCP 风格工具
│   ├── benchmarks/       # 确定性评测
│   ├── observability/    # trace、AgentRun、成本追踪
│   ├── data_adapters/    # 外部数据集 ETL
│   └── common/           # 配置、DB、LLM/Embedding 工厂
├── scripts/              # seed、smoke test、数据管线
├── infra/                # docker-compose.yml
├── data/                 # seed、eval、raw/processed（本地）
├── assets/               # 架构图等资源
└── Makefile
```

---

## 权限模型（摘要）

| 角色 | 典型能力 |
|------|----------|
| admin / executive | 全权限 + Benchmark |
| procurement_manager | 上传文档、supplier_impact |
| logistics_manager | delivery_delay_impact |
| quality_manager | quality_issue_trace |
| external_supplier | 仅本供应商 scope |

密级：`public(0) < internal(1) < department(2) < confidential(3) < executive(4)`

---

## Makefile 常用命令

| 命令 | 说明 |
|------|------|
| `make up` | 已有镜像直接复用，缺少镜像时通过 crane 拉取，然后启动 Compose |
| `make pull-crane` | 用 crane 强制刷新全部 Compose 镜像并加载到 Docker |
| `make migrate` / `make seed` | 数据库迁移与种子数据 |
| `make api` / `make web` | 启动后端 / 前端 |
| `make build-graph` | 构建 Neo4j 演示图谱 |
| `make ingest-seed-docs` | Seed 文档入库 PG + Qdrant |
| `make bench` | 运行 Benchmark（limit 100） |
| `make smoke-mvp3` | Secure RAG 端到端冒烟 |
| `uv run pytest tests/test_planning.py` | Planner / Validator / DAG 执行单元测试 |
| `uv run python scripts/run_agent_eval_v2.py` | 运行 150 条 Agent Plan V2 端到端评测 |

---

## 当前范围与限制

**已实现：**

- Secure RAG + GraphRAG + Memory + MCP Tools + LangGraph Chat
- Planner V1：规则/LLM 双路径规划、最多 8 个 RAG/Graph/Tool/Memory 任务、DAG 依赖与同层并行
- 计划级校验：Schema、循环依赖、实体真实性、租户边界、角色权限及高风险审批
- 计划与任务运行持久化；Chat API 和 Trace 页面展示规划来源、置信度与逐任务状态
- JWT + RBAC + 三层权限（Qdrant / Graph / Tool）
- Next.js 全功能控制台 + Benchmark + Trace 回放
- 111 条基础 Benchmark + 150 条 Agent Plan V2 Benchmark；全链路 `trace_id` 可观测

**尚未实现：**

- Text2Cypher、Graph Community Summary、Cross-Encoder Rerank
- RAGAS / LLM-as-Judge
- MinIO 主存储（仅健康检查预留）
- 真实 ERP 实时对接

---

## License

[MIT](LICENSE)
