# ML Infrastructure Engineer — Interview Process & Preparation Guide
*May 2026*

---

## Overview

ML infra roles occupy a narrow technical band between systems engineering and ML research. The interview loop is harder to prepare for than a standard SWE loop because it blends low-level systems knowledge (kernels, distributed computing, GPU memory hierarchies) with ML depth (training dynamics, inference optimization, RL systems). The bar has risen significantly at frontier labs in 2025–2026, and several companies now run dedicated **ML infra system design** rounds separate from generic distributed systems rounds.

---

## Typical Interview Loop (5–7 rounds total)

### Stage 1 — Recruiter / Hiring Manager Screen (30–45 min)
- Career narrative: why ML infra, what you've built, scale of systems
- Logistics: level, compensation expectations, location, timeline
- Some HMs ask a light technical signal question here

**What to prepare:** A crisp 3-minute story covering your most impactful ML infra project — what the problem was, your technical approach, and the measurable outcome (latency reduction %, training throughput improvement, etc.).

---

### Stage 2 — Technical Phone Screen (45–60 min)
Usually one of:
- **Coding**: Medium–Hard LeetCode problems focused on graphs, queues, concurrency, or systems primitives. At frontier labs (xAI, Anthropic, OAI), expect distributed systems coding — e.g., a distributed matrix multiplication problem.
- **ML fundamentals discussion**: walk through a training pipeline, describe FSDP vs. tensor parallelism trade-offs, explain how you'd debug a training run that isn't converging.

**What to prepare:** Stay sharp on PyTorch internals, GPU memory layout, collective communication primitives (all-reduce, all-gather), and standard LeetCode patterns.

---

### Stage 3 — Coding Round(s) (1–2 rounds, 45–60 min each)

**What's tested:**
- Standard algorithmic problems (Medium/Hard LeetCode) — graphs, DP, two pointers
- **ML-flavored coding**: implement a custom CUDA/Triton kernel, write a batched attention function in NumPy, debug a broken PyTorch training loop, implement gradient checkpointing logic
- **Object-oriented systems coding** (common at Netflix, Snap, Reddit): implement a mini KV store, a rate limiter, a job scheduler, a feature store interface — starting simple and scaling up

**What to prepare:**
- Solve 100–150 LeetCode problems; weight toward graph BFS/DFS, sliding window, heaps
- Practice implementing attention, softmax, layer norm, and simple distributed collectives from scratch in PyTorch/NumPy
- Read [GPU MODE lectures](https://github.com/gpu-mode/lectures) for kernel-level intuition

---

### Stage 4 — ML System Design (60 min)

**What's tested:**  
Design an end-to-end ML system — e.g., "design a recommendation system for Instagram Reels," "design a real-time content moderation pipeline," "design a distributed training system for a 70B parameter model."

**Framework (6 steps):**
1. **Clarify requirements** — ask about scale, latency SLOs, freshness requirements, budget
2. **Data pipeline** — ingestion, preprocessing, feature engineering, storage (feature store design)
3. **Model design** — architecture choice, trade-offs vs. alternatives
4. **Training infrastructure** — distributed strategy (data/tensor/pipeline parallelism), experiment tracking, checkpointing
5. **Serving/inference** — online vs. batch, latency budgets, batching strategies, caching, fallback behavior
6. **Monitoring & iteration** — drift detection, A/B testing, retraining triggers

**Key trade-offs to discuss:** accuracy vs. latency, real-time vs. batch inference, GPU cost vs. throughput, model freshness vs. serving stability.

**What to prepare:** Read the ML system design chapter in *Designing Machine Learning Systems* (Chip Huyen). Study engineering blogs from Uber, Doordash, Meta, Netflix on their ML platforms.

---

### Stage 5 — ML Infra System Design (60 min) *(Required at: Netflix, Snap, Reddit, Doordash, Notion; common at frontier labs)*

This is **not** ML system design — it ignores modeling entirely and focuses on the infrastructure layer:

**Example questions:**
- "Design a distributed training framework that handles fault tolerance at the 1,000-GPU scale"
- "Design a feature store that supports both online (low-latency) and offline (batch) feature serving with point-in-time correctness"
- "Design an inference serving system that supports multi-model batching, dynamic batching, and SLA-based priority queuing"
- "How would you design a hyperparameter tuning system for large-scale distributed training?"

**Key subsystems to know deeply:**
- **Training infra**: job schedulers (SLURM, K8s), collective communication (NCCL), checkpointing strategies, failure recovery, gradient compression
- **Feature stores**: online (Redis/DynamoDB) vs. offline (Iceberg/Parquet) paths, point-in-time joins to prevent leakage, streaming aggregations via Kafka + Flink
- **Inference serving**: vLLM/TGI architecture, continuous batching, KV cache management, speculative decoding, disaggregated prefill/decode
- **Observability**: GPU utilization tracking, training loss anomaly detection, latency percentile monitoring

**What to prepare:** Read Yuan Meng's [ML Infra System Design blog post](https://www.yuan-meng.com/posts/ml_infra_interviews/). Study vLLM's architecture, Megatron-LM's parallelism docs, and the Tecton/Feast feature store documentation.

---

### Stage 6 — Behavioral / Leadership (30–45 min)

**What's tested:** Ownership, cross-functional collaboration, handling ambiguity, influencing without authority, handling production incidents.

**Common questions:**
- "Tell me about a time you made an infrastructure decision that turned out to be wrong — how did you recover?"
- "Describe a situation where you had to push back on a researcher's request because it was infeasible at scale"
- "Walk me through a large-scale on-call incident you owned — what was your debugging process?"
- "How have you made your team's ML systems more reliable without adding toil?"

**STAR format** — Situation, Task, Action, Result. Prepare 5–6 stories that each cover multiple dimensions (technical depth, collaboration, ownership, scale).

---

### Stage 7 — Deep Dive / Technical Bar Raiser (45–60 min)

Common at Google, Amazon, Anthropic, and frontier labs. An experienced senior engineer probes the depth of your claimed expertise:

- **Kernel-level questions**: walk through how you'd fuse an attention kernel, explain warp-level primitives, describe memory coalescing
- **Distributed training deep dive**: explain the exact communication pattern in ring-all-reduce, when to prefer FSDP over Megatron pipeline parallelism, how to handle gradient stalls
- **Inference optimization**: explain continuous batching vs. static batching, PagedAttention memory management, speculative decoding prefill/verify steps

---

## Preparation Tips by Topic

### Systems & Infrastructure
- **Read**: *Computer Systems: A Programmer's Perspective* (Ch. 6 memory hierarchy), CUDA Programming Guide
- **Practice**: implement a basic CUDA matmul kernel; profile it with Nsight; optimize memory access patterns
- **Know cold**: GPU memory hierarchy (registers → shared → L2 → HBM), warp execution model, PCIe vs. NVLink bandwidth numbers

### Distributed Training
- **Know cold**: data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism, FSDP sharding strategies
- **Read**: Megatron-LM paper, PyTorch FSDP docs, DeepSpeed ZeRO paper
- **Practice**: explain from first principles what happens to gradients in ring-all-reduce with N GPUs

### Inference Optimization
- **Know cold**: KV cache, continuous batching, speculative decoding, quantization (GPTQ, AWQ, FP8), paged attention
- **Read**: vLLM paper, TRT-LLM docs, the PagedAttention paper
- **Practice**: estimate the memory footprint of a 70B model in FP16 served with a batch size of 32

### Coding
- LeetCode: target 150+ problems, focus on hard graph problems, system design coding (OOP)
- ML coding: be ready to implement forward/backward pass for attention, custom PyTorch autograd function, a basic tensor parallel linear layer

### ML Fundamentals
- Transformers architecture (attention, positional encoding, layer norm variants)
- Training dynamics: gradient flow, loss spikes, learning rate schedules, mixed precision (BF16/FP8)
- RL for LLMs: RLHF/GRPO/PPO pipeline, reward model training, KL penalty

---

## Company-Specific Notes

| Company | Interview Style | Notable Rounds |
|---|---|---|
| Anthropic | Practical, deep systems; values written communication | Coding + systems design; may include a take-home |
| OAI | High bar on distributed systems + perf profiling | Deep technical dives; GPU kernel questions |
| xAI | Aggressive bar; fast-paced; 2 rounds seen in 2026 | Distributed matmul coding + deep dive with live coding |
| Netflix | $500–900K total comp; very high bar | Coding + ML system design + **ML infra system design** |
| Snap / Reddit / DoorDash | Standard MLE loop + **dedicated ML infra round** | ML infra design is a required, separate round |
| Meta | Applied ML focus; no PhD required | Coding + ML system design + behavioral; no ML infra design as separate round |
| Google | In-person onsite (as of early 2026) | Coding + ML system design + Googleyness |
| Jane Street | Very low-level; kernel/GPU focus | Domain-specific; C++ kernel questions |

---

## Recommended Resources

**Books**
- *Designing Machine Learning Systems* — Chip Huyen
- *System Design Interview* Vols 1 & 2 — Alex Xu
- *Database Internals* — Alex Petrov (for feature store internals)

**Online**
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) — CUDA/Triton kernel programming
- [Yuan Meng's ML Infra System Design post](https://www.yuan-meng.com/posts/ml_infra_interviews/) — best single resource for the infra design round
- [Yuan Meng's MLE 2.0 post](https://www.yuan-meng.com/posts/mle_interviews_2.0/) — 2025/2026 interview landscape overview
- [LLM Systems reading list](https://github.com/HazyResearch/fly) — papers on inference and training optimization
- vLLM, Megatron-LM, DeepSpeed GitHub repos — read the architecture docs

**Papers to know**
- Attention Is All You Need (transformer architecture)
- FlashAttention 1 & 2 (memory-efficient attention)
- PagedAttention / vLLM (KV cache management)
- DeepSpeed ZeRO (optimizer state sharding)
- Megatron-LM (tensor + pipeline parallelism)
- Speculative Decoding (Leviathan et al.)
