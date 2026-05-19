# 50 Top Interview Questions for ML Infrastructure Roles

A curated list of interview questions for ML infrastructure, training, inference, and systems engineering positions based on common ML infra job requirements.

---

## 🔧 GPU & Hardware Fundamentals (Questions 1-6)

1. **Explain the memory hierarchy in NVIDIA GPUs and how it affects kernel performance optimization.** (Memory bandwidth, cache coherency, warp scheduling)
   - Key topics: Global, shared, constant, texture memory; memory coalescing; occupancy

2. **What is a warp and how do thread blocks interact during execution?**
   - Key concepts: 32 threads per warp, SIMD model, branch divergence, occupancy calculation

3. **How would you optimize a kernel for memory bandwidth vs. compute? Give an example.**
   - Discuss: Arithmetic intensity, roofline model, memory-bound vs. compute-bound kernels

4. **Explain tensor core operations and how to utilize them in matrix multiplications.**
   - Topics: Mixed precision, FP16/BF16 operations, CUTLASS library, optimization techniques

5. **What are the performance implications of using different data types (FP32, FP16, INT8) in deep learning?**
   - Quantization, precision loss, numerical stability, hardware support

6. **How do you profile and benchmark GPU kernels? What tools do you use?**
   - NVIDIA Nsight, nvprof, cutlass profiler, measuring throughput and latency

---

## 🏃 Model Inference Optimization (Questions 7-13)

7. **Describe the inference pipeline for serving an LLM and identify bottlenecks at each stage.**
   - Tokenization → Prefill → Token generation → Decoding; KV cache management

8. **What is speculative decoding and how does it improve inference latency?**
   - Draft model, acceptance/rejection, draft tokens, latency reduction strategies

9. **Explain KV cache optimization techniques and trade-offs (e.g., quantization, pruning, eviction policies).**
   - Memory footprint, computational cost, accuracy impact, PagedAttention

10. **How does batching affect inference throughput and what challenges arise with dynamic batch sizes?**
    - Token batching, continuous batching, request scheduling, padding overhead

11. **Design a system to serve multiple LLM variants with different precision and batch sizes.**
    - Resource allocation, request routing, model management, SLO handling

12. **What strategies would you use to optimize prefill vs. token generation phases separately?**
    - Different memory patterns, compute requirements, pipelining techniques

13. **Explain the differences between static and dynamic batching and when each is appropriate.**
    - Latency vs. throughput trade-offs, hardware utilization, request patterns

---

## 🎓 Training & Distributed Systems (Questions 14-21)

14. **How does data parallelism differ from model parallelism and tensor parallelism?**
    - When to use each, gradient aggregation, communication patterns, scaling limits

15. **Explain pipeline parallelism and how gradient bubbles affect training efficiency.**
    - Stage synchronization, microbatch scheduling, efficiency metrics

16. **Design a distributed training system for a 70B parameter model across 32 GPUs.**
    - Hardware topology, parallelism strategy, communication overhead, optimization

17. **What is gradient checkpointing and what are the computation vs. memory trade-offs?**
    - Activation recomputation, memory savings, computational overhead, best practices

18. **How do you handle stragglers and synchronization in distributed training?**
    - Heterogeneous hardware, network bottlenecks, asynchronous methods, fault tolerance

19. **Explain ring all-reduce and compare it with tree-based collective communication.**
    - Network bandwidth utilization, latency sensitivity, implementation details

20. **How would you implement efficient distributed RL training with policy rollouts and gradients?**
    - Actor-critic architectures, experience collection, gradient updates, synchronization

21. **What monitoring and debugging tools do you use for training systems at scale?**
    - Profiling training bottlenecks, identifying communication vs. computation time, optimization

---

## 🔄 Compilation & Framework Optimization (Questions 22-28)

22. **How do modern compilers (PyTorch Compile, XLA) optimize computation graphs?**
    - Graph fusion, memory planning, kernel selection, compilation overhead

23. **Explain how PyTorch FSDP (Fully Sharded Data Parallel) works and its communication patterns.**
    - Sharding strategies, communication phases, memory efficiency, overlapping communication

24. **What is the difference between eager execution and graph mode? When should each be used?**
    - PyTorch vs. TensorFlow paradigms, tradeoffs, debugging, production deployment

25. **How would you optimize a custom PyTorch operation for a specific GPU architecture?**
    - CUDA kernels, fusion with neighboring ops, mixed precision, deployment

26. **Explain automatic mixed precision (AMP) and how it affects training stability.**
    - Loss scaling, gradient overflow, convergence, performance gains

27. **How does JAX's functional programming model enable better compilation and parallelization?**
    - Pure functions, pmap, vmap, compilation, autodiff compatibility

28. **Design a system to compile and optimize multiple different model architectures efficiently.**
    - Caching, dynamic shapes, operator fusion strategies, memory optimization

---

## 🌐 Distributed Communication (Questions 29-33)

29. **Explain NCCL (NVIDIA Collective Communications Library) and its role in distributed training.**
    - Collective operations, topology awareness, bandwidth optimization, fault tolerance

30. **How does overlap of communication and computation improve training throughput?**
    - Gradient computation pipelining, communication scheduling, memory management

31. **Design a communication strategy for training across multiple nodes with heterogeneous network bandwidth.**
    - Adaptive strategies, compression, priority, batching

32. **What are the trade-offs between MPI, NCCL, and gRPC for distributed ML systems?**
    - Latency, throughput, fault tolerance, programming complexity

33. **How would you implement efficient gradient compression for communication-constrained environments?**
    - Quantization, sparsification, error feedback, convergence guarantees

---

## 💾 Systems & Infrastructure (Questions 34-40)

34. **Design a containerized ML training platform supporting GPU isolation and resource management.**
    - Kubernetes, scheduling, resource allocation, multi-tenant fairness

35. **How would you implement fault tolerance for long-running training jobs?**
    - Checkpointing strategies, checkpoint storage, recovery mechanisms, data consistency

36. **Explain effective caching strategies for model weights, activations, and gradients.**
    - Eviction policies, LRU variations, bandwidth optimization, memory hierarchies

37. **How do you monitor and optimize cluster utilization for ML workloads?**
    - Metrics, bottleneck identification, resource forecasting, cost optimization

38. **Design a system to support both training and inference workloads on shared GPU clusters.**
    - Resource isolation, priority handling, SLO management, cost allocation

39. **What database and caching technologies would you use to manage model artifacts at scale?**
    - Model versioning, metadata, distributed caching, serialization formats

40. **How would you implement efficient data loading and preprocessing for large-scale training?**
    - Data pipelines, prefetching, augmentation, format optimization, storage backends

---

## 🧠 Advanced ML Systems Topics (Questions 41-46)

41. **Explain speculative decoding, lookahead decoding, and other novel inference optimization techniques.**
    - Theoretical foundations, implementation trade-offs, when each is beneficial

42. **How do you implement dynamic shape handling in compiled ML systems?**
    - Shape inference, kernel dispatch, memory allocation, performance implications

43. **Design a post-training infrastructure for RLHF (Reinforcement Learning from Human Feedback).**
    - Data collection, reward models, PPO training, sampling efficiency, scaling

44. **Explain the architecture and optimization challenges in mixture-of-experts (MoE) models.**
    - Expert routing, load balancing, communication patterns, computational efficiency

45. **How would you optimize a flash attention implementation and explain its advantages?**
    - IO-awareness, memory access patterns, backward pass, hardware support

46. **Design a system for efficient multi-modal model serving (images, text, video) with shared infrastructure.**
    - Feature processing, tensor operations, memory management, latency SLOs

---

## 🎯 Performance Optimization & Problem-Solving (Questions 47-50)

47. **Your inference service has 1000ms p99 latency instead of target 100ms. Walk through your debugging approach.**
    - Profiling methodology, identifying bottlenecks, optimization priorities, measurement validation

48. **Design an ML profiling tool that identifies bottlenecks in a training run across compute, communication, and I/O.**
    - Metric collection, visualization, root cause analysis, automated recommendations

49. **You need to reduce training time from 30 days to 10 days. What optimizations would you explore systematically?**
    - Hardware utilization, communication overlap, computation efficiency, batch size tuning

50. **Design a parameter-efficient fine-tuning system (e.g., LoRA) that maintains inference performance while reducing memory.**
    - Architecture modifications, training efficiency, inference overhead, quality-performance trade-offs

---

## 📚 Additional Topic Areas to Prepare

### Kernels & CUDA Programming
- Warp-level primitives and their use cases
- Bank conflicts and memory access patterns
- Optimization with CUTLASS templates
- Triton programming for custom operations

### Quantization & Compression
- Post-training quantization (INT8, INT4)
- Quantization-aware training
- Knowledge distillation strategies
- Trade-offs between latency, memory, and accuracy

### System Design Questions
- Designing end-to-end ML pipelines
- Managing multiple models in production
- Scaling strategies for different bottlenecks
- Cost vs. performance optimization

### Reinforcement Learning Systems
- Experience replay buffers and sampling
- Policy gradient computation at scale
- Multi-agent training infrastructure
- Curriculum learning systems

### Research & Implementation
- Reading and implementing research papers
- Novel optimization techniques
- Measuring statistical significance
- Balancing innovation with stability

---

## 🎓 Study Resources (from your materials list)

### Core Learning
- **CUDA Programming**: PMPP book, NVIDIA official documentation, Oxford CUDA course
- **GPU Architecture**: NVIDIA blogs, Modal GPU glossary, Lei Mao's CUDA blog
- **Distributed Systems**: Designing Data-Intensive Applications (DDIA), System Design Primer
- **ML Systems**: Stanford MLSys course, JAX scaling book

### Frameworks & Tools
- **PyTorch**: Official docs, PyTorch blog, TorchTitan examples
- **JAX**: JAX scaling book, official tutorials
- **Inference**: Modal LLM almanac, vLLM documentation
- **Compilation**: JAX XLA, PyTorch Compile, TVM documentation

### Technical Blogs & Research
- Anthropic Engineering Blog
- OpenAI Engineering Blog
- Together AI Blog
- Meta Research & Engineering
- ByteDance Research Blog
- NVIDIA Technical Blog

---

## 💡 Interview Tips

1. **Ask clarifying questions** before diving into solutions
2. **Focus on trade-offs** - performance vs. memory, latency vs. throughput, etc.
3. **Show your reasoning** - explain why you'd choose one approach over another
4. **Discuss real constraints** - scale, hardware, budget, power consumption
5. **Be prepared to code** - be ready to write CUDA, PyTorch, or system design code
6. **Benchmark and measure** - always quantify improvements with real numbers
7. **Think about edge cases** - different hardware, batch sizes, sequence lengths
8. **Stay updated** - follow ML systems research and engineering blogs
