
Created: 07-07-2025 15:35

Tags: [[Generative AI]] [[20_ZETTLEKASTEN/Reinforcement Learning]]

Status: #baby 

## References

1. https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/
2. https://rail.eecs.berkeley.edu/deeprlcourse/
3. https://stanford-cs336.github.io/spring2025/#schedule
4. https://sites.google.com/view/cjin/teaching/ece524
5. https://web.stanford.edu/class/cs234/index.html

Use this method: [[Knowledge Forging Technique]]
## **Simplified Structure: 1-2 Posts Per Week**

**Study Schedule**: 
- **Monday-Wednesday**: Study + Code + Detailed Notes (4-5 hours total)
- **Thursday**: Review agent-generated blog + polish (30-45 minutes)
- **Friday**: Optional: Second topic or deeper experiments
- **Weekend**: Advanced experiments or next week prep

**Blog Schedule**: 
- **2 blogs per week** (now easily achievable with agent)
- **1 primary topic + 1 connection/experiment post** per week
- **Agent handles draft, you focus on technical accuracy and insights**

---
## Editions: 2 Types of Audience

- Scientist Edition
- Manager or Non-tech edition

---

## **Phase 1: Core Foundations (Weeks 1-4)**

### **Week 1: RL & MDP Fundamentals**

**Study Topics:**
- Markov Decision Processes (states, actions, rewards, transitions)
- Bellman equations and dynamic programming
- Value iteration and policy iteration
- **Core Papers**: Sutton & Barto Ch 3-4

**Experiments:**
- Simple grid world MDP solver with multiple environments
- Policy vs value iteration comparison study
- **WandB**: Convergence curves, value function heatmaps, algorithm comparison


Blog 1: 

**Blog 1**: "Decision Making 101: How AI Learns to Choose"
**Notes for Agent**: Start with everyday decision examples (route planning, career choices). Formalize with MDP mathematical framework. Include visual grid world examples and value iteration code walkthrough. Emphasize intuition behind mathematical concepts.

**Blog 2**: "The Mathematics of Optimal Decisions: Bellman Equations Explained"
**Notes for Agent**: Deep dive into Bellman optimality equations. Show mathematical derivations step-by-step. Connect to dynamic programming principles. Include code comparing value vs policy iteration with performance metrics.

---

### **Week 2: Tokenization Deep Dive**
**Study Topics:**
- Byte Pair Encoding (BPE) algorithm implementation details
- WordPiece and SentencePiece variations and trade-offs
- Vocabulary construction optimization and compression principles
- Information theory connections (entropy, compression)
- **Core Papers**: Sennrich BPE paper, SentencePiece paper

**Experiments:**
- Build BPE tokenizer from scratch with multiple merge strategies
- Compare compression ratios across languages and domains
- Analyze vocabulary efficiency vs. coverage trade-offs
- **WandB**: Tokenization metrics, vocabulary size effects, cross-lingual analysis

**Blog 1**: "Breaking Down Language: How AI Reads Text"
**Notes for Agent**: Start with "how do you teach a computer to read?" problem. Build BPE algorithm step by step with clear examples. Show compression concept with visual examples. Include complete code implementation walkthrough. Connect to information theory basics.

**Blog 2**: "The Tokenization Wars: BPE vs WordPiece vs SentencePiece"
**Notes for Agent**: Comparative analysis of different tokenization methods. Include experimental results from WandB showing trade-offs. Discuss when to use each method. Connect to practical LLM deployment considerations. Reference compression principles from Blog 1.

---

### **Week 3: Attention Mechanisms - Complete Landscape**
**Study Topics:**
- **Basic Attention**: Scaled dot-product, additive attention, multiplicative attention
- **Advanced Attention Variants**: 
  - Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
  - Flash Attention, Flash Attention 2, Memory-efficient attention
  - Sliding window attention, Local attention patterns
  - Sparse attention (BigBird, Longformer patterns)
  - Linear attention approximations (Performer, Linformer)
- **Positional Encodings**: 
  - Sinusoidal, learned absolute positions
  - Relative Position Representations (Shaw et al.)
  - Rotary Position Embeddings (RoPE)
  - ALiBi (Attention with Linear Biases)
  - Complex-valued rotary embeddings
- **Attention Analysis**:
  - Attention head analysis and pruning
  - Attention entropy and distribution analysis
  - Multi-scale attention patterns
- **Core Papers**: 
  - "Attention Is All You Need", "Self-Attention with Relative Position Representations"
  - "FlashAttention: Fast and Memory-Efficient Exact Attention", "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  - "Train Short, Test Long: Attention with Linear Biases (ALiBi)"

**Experiments:**
- Implement all major attention variants from scratch
- Memory and speed benchmarking across attention types
- Attention pattern visualization and analysis toolkit
- Position encoding comparison on length extrapolation tasks
- **WandB**: Memory usage, speed comparisons, attention entropy, extrapolation performance

**Blog 1**: "The Evolution of Attention: From Dot-Product to Flash Attention"
**Notes for Agent**: Start with basic attention mechanism from Week 2 context. Build complexity progressively through MQA, GQA, Flash Attention. Show memory/speed trade-offs with benchmarking results. Include implementation details and optimization tricks.

**Blog 2**: "Position, Position, Position: The Hidden Battle in Transformers"
**Notes for Agent**: Deep dive into positional encoding methods. Show length extrapolation experiments. Compare RoPE vs ALiBi vs sinusoidal with concrete results. Connect to real-world deployment challenges. Include visualization of position representations.

---

### **Week 4: Policy Optimization - The Complete Arsenal**
**Study Topics:**
- **Classic Policy Gradients**: REINFORCE, Williams 1992
- **Natural Policy Gradients**: Natural gradients, Fisher information matrix
- **Trust Region Methods**: 
  - TRPO (Trust Region Policy Optimization)
  - CPO (Constrained Policy Optimization)
  - PCPO (Primal-Dual Constrained Policy Optimization)
- **Proximal Methods**: 
  - PPO (Proximal Policy Optimization) - clipped and adaptive KL variants
  - APPO (Asynchronous PPO), IMPALA
- **Advanced Policy Optimization**:
  - GRPO (Group Relative Policy Optimization) - latest for RLHF
  - DPO (Direct Preference Optimization) - recent breakthrough
  - IPO (Identity Preference Optimization)
  - SimPO (Simple Preference Optimization)
  - KTO (Kahneman-Tversky Optimization)
- **Variance Reduction**:
  - Control variates, importance sampling corrections
  - Off-policy corrections (IMPALA-style V-trace)
  - Generalized Advantage Estimation (GAE)
- **Core Papers**: 
  - Williams REINFORCE, Schulman TRPO/PPO papers
  - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  - "Group Relative Policy Optimization for RLHF"
  - "KTO: Model Alignment as Prospect Theory"

**Experiments:**
- Implement full policy optimization suite (REINFORCE → TRPO → PPO → GRPO)
- Comparative study on multiple environments
- Variance reduction technique ablations
- Preference optimization comparison (DPO vs IPO vs SimPO)
- **WandB**: Learning curves, variance analysis, constraint satisfaction, preference accuracy

**Blog 1**: "The Policy Gradient Wars: From REINFORCE to GRPO"
**Notes for Agent**: Chronicle the evolution of policy optimization. Start with REINFORCE variance problems, show how each method (TRPO, PPO, GRPO) solves specific issues. Include mathematical derivations and implementation details. Show experimental comparisons.

**Blog 2**: "Beyond PPO: The New Generation of Preference Optimization"
**Notes for Agent**: Focus on recent breakthroughs in preference optimization. Explain DPO's breakthrough insight, compare with IPO, SimPO, KTO. Show how these avoid reward modeling. Include experimental results and practical implications for LLM training.

---

## **Phase 2: Advanced Methods (Weeks 5-8)**

### **Week 5: Transformer Architecture - Beyond the Basics**
**Study Topics:**
- **Core Architecture**: Layer norm placement, residual connections, feed-forward variants
- **Normalization Innovations**:
  - Pre-norm vs Post-norm (critical for training stability)
  - RMSNorm, LayerNorm variants
  - QK-Norm for attention stability
- **Activation Functions**: 
  - SwiGLU (used in LLaMA), GeGLU, ReLU variants
  - Gated Linear Units and their impact
- **Advanced Architectures**:
  - **LLaMA architecture**: RMSNorm, SwiGLU, RoPE integration
  - **PaLM architecture**: Parallel attention and FFN blocks
  - **GPT-4 architecture**: Sparse MoE rumors and analysis
  - **Mamba/State Space Models**: Alternative to attention
  - **RetNet**: Alternative with better scaling properties
  - **Mamba-2**: Latest state space model developments
- **Efficiency Innovations**:
  - **Mixture of Experts (MoE)**: Switch Transformer, GLaM, PaLM-2
  - **Sparse attention patterns**: Fixed patterns, learned patterns
  - **Gradient checkpointing** and memory optimization
  - **Mixed precision training** (FP16, BF16, FP8)
- **Architectural Scaling**:
  - **Parallel attention + FFN** vs sequential
  - **Width vs depth scaling** laws
  - **Architectural ablation studies**
- **Core Papers**: 
  - "LLaMA: Open and Efficient Foundation Language Models"
  - "PaLM: Scaling Language Modeling with Pathways"
  - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  - "Switch Transformer: Scaling to Trillion Parameter Models"

**Experiments:**
- Build modular transformer with swappable components
- Architecture ablation study (norm placement, activations, etc.)
- MoE implementation and routing analysis
- Memory and compute profiling across architectures
- **WandB**: Training stability, memory usage, routing efficiency, scaling curves

**Blog 1**: "Transformer Architecture Evolution: From Attention to Mamba"
**Notes for Agent**: Trace architectural evolution from original Transformer through LLaMA, PaLM, to Mamba. Show key innovations and their motivations. Include detailed architecture diagrams and implementation details. Focus on training stability improvements.

**Blog 2**: "The Efficiency Revolution: MoE, Sparse Attention, and Beyond"
**Notes for Agent**: Deep dive into efficiency innovations. Explain MoE routing, sparse attention patterns, memory optimization techniques. Include experimental results showing scaling benefits. Connect to practical deployment considerations.

---

### **Week 6: Actor-Critic Methods**
**Study Topics:**
- TD learning and bootstrapping
- Advantage functions
- A2C algorithm
- **Core Papers**: A2C paper, TD learning basics

**Experiments:**
- Implement A2C on Atari
- Compare with REINFORCE from Week 4
- **WandB**: Stability comparisons, learning efficiency

**Blog**: "The Best of Both Worlds: Actor-Critic Methods"
**Content**: Reference policy gradients from Week 4 and value functions from Week 1. Show how actor-critic combines benefits. Clear implementation.

---

### **Week 7: PPO & Trust Regions**
**Study Topics:**
- Trust region theory
- PPO clipped objective
- Importance sampling
- **Core Papers**: PPO paper, TRPO paper

**Experiments:**
- Implement PPO with clipping
- Compare constraint satisfaction
- **WandB**: Clipping ratios, KL divergence tracking

**Blog**: "PPO: The Goldilocks of Policy Optimization"
**Content**: Reference actor-critic from Week 6. Show evolution from REINFORCE → A2C → PPO. Explain why PPO became standard. Clear implementation.

---

### **Week 8: Language Modeling**
**Study Topics:**
- Autoregressive generation
- Cross-entropy loss and perplexity
- Teacher forcing and exposure bias
- **Core Papers**: GPT-1 paper, language modeling basics

**Experiments:**
- Train small autoregressive transformer
- Study perplexity vs model size
- **WandB**: Training curves, generation quality

**Blog**: "Teaching Machines to Speak: The Art of Language Modeling"
**Content**: Combine transformer architecture (Week 5) with tokenization (Week 2). Show autoregressive generation process. Connect to sequential decision making from Week 1.

---

## **Phase 3: RLHF & Safety (Weeks 9-12)**

### **Week 9: Human Preference Learning - Complete Landscape**
**Study Topics:**
- **Preference Modeling Foundations**:
  - Bradley-Terry models, Plackett-Luce models
  - Elo rating systems for AI evaluation
  - Pairwise vs listwise preference learning
- **Advanced Preference Collection**:
  - Constitutional preference generation
  - Active learning for preference elicitation  
  - Preference uncertainty quantification
  - Multi-objective preference learning
- **Reward Model Architectures**:
  - Architecture choices (shared vs separate towers)
  - Ensemble reward models for uncertainty
  - Preference model scaling laws
  - Reward model robustness techniques
- **Latest RLHF Variants**:
  - **RLHF-V**: Value-based RLHF variants
  - **Constitutional AI**: AI feedback loops
  - **RLAIF**: Reinforcement Learning from AI Feedback
  - **Self-Rewarding Language Models**: Models that improve their own rewards
- **Preference Dataset Engineering**:
  - HH-RLHF dataset analysis
  - Anthropic's Constitutional AI datasets
  - Preference data quality and bias analysis
  - Synthetic preference generation techniques
- **Core Papers**: 
  - "Training language models to follow instructions with human feedback" (InstructGPT)
  - "Constitutional AI: Harmlessness from AI Feedback"
  - "Self-Rewarding Language Models"
  - "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"

**Experiments:**
- Implement multiple preference learning algorithms
- Reward model uncertainty analysis and ensemble methods
- Preference dataset quality analysis toolkit
- Constitutional AI preference generation pipeline
- **WandB**: Preference accuracy, reward calibration, uncertainty metrics, constitutional training

**Blog 1**: "The Science of Preference: From Human Feedback to AI Alignment"
**Notes for Agent**: Start with preference learning fundamentals, build through Bradley-Terry to latest Constitutional AI methods. Show evolution from simple pairwise preferences to complex constitutional feedback. Include mathematical foundations and practical implementation.

**Blog 2**: "Building Better Reward Models: Architecture, Uncertainty, and Robustness"
**Notes for Agent**: Technical deep-dive into reward model design. Cover architecture choices, ensemble methods, uncertainty quantification. Include experimental analysis of different approaches. Connect to downstream RLHF performance and safety considerations.

---

### **Week 10: PPO for Language Models**
**Study Topics:**
- Adapting PPO for text generation
- KL penalties and reward-performance tradeoffs
- Sequence-level rewards
- **Core Papers**: InstructGPT technical details

**Experiments:**
- Implement PPO for text generation
- Study KL-reward tradeoffs
- **WandB**: RLHF training dynamics

**Blog**: "RLHF in Practice: PPO Meets Language Models"
**Content**: Combine PPO from Week 7 with language modeling from Week 8 and preferences from Week 9. Show technical challenges of discrete optimization.

---

### **Week 11: Constitutional AI & Advanced Safety Research**
**Study Topics:**
- **Constitutional AI Deep Dive**:
  - Constitutional principles design and optimization
  - Multi-turn constitutional training
  - Constitutional chain-of-thought reasoning
  - Scalable oversight via constitutional methods
- **Advanced AI Safety Research**:
  - **Weak-to-Strong Generalization**: Training strong models with weak supervisors
  - **Superalignment research**: Aligning models more capable than humans
  - **Interpretability advances**: Sparse autoencoders, feature visualization
  - **Activation Patching**: Understanding model internals through intervention
- **Latest Safety Techniques**:
  - **Process vs Outcome supervision**: Latest research on training process
  - **AI Safety via debate**: Multi-agent debate for alignment
  - **Scalable oversight**: Methods that scale beyond human evaluation
  - **Red team analysis**: Advanced adversarial testing methods
- **Safety Evaluation Frameworks**:
  - **Model evaluations**: Capability and safety assessments
  - **Dangerous capability detection**: Early warning systems
  - **Alignment evaluation**: Measuring alignment robustness
  - **Safety benchmarks**: Latest evaluation suites and metrics
- **Anthropic-Specific Research**:
  - **Claude training methodology**: Latest insights from Anthropic papers
  - **Harmlessness vs helpfulness**: Balancing safety and capability
  - **Transparency research**: Model cards, system cards, evaluation
- **Core Papers**: 
  - "Constitutional AI: Harmlessness from AI Feedback"
  - "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision"
  - "AI Safety via Debate", "Scalable Oversight of AI Systems"
  - Latest Anthropic safety research papers

**Experiments:**
- Implement full constitutional AI training pipeline
- Weak-to-strong generalization experiments
- Interpretability analysis using sparse autoencoders
- Red team evaluation framework development
- **WandB**: Constitutional training metrics, weak-to-strong performance, interpretability visualizations

**Blog 1**: "Constitutional AI: Teaching Machines Right from Wrong"
**Notes for Agent**: Build on RLHF from previous weeks. Deep dive into constitutional training methodology. Show how AI feedback scales beyond human feedback. Include implementation details and experimental results. Connect to broader alignment research.

**Blog 2**: "The Alignment Challenge: From Weak-to-Strong Generalization to Superalignment"  
**Notes for Agent**: Cover cutting-edge alignment research. Explain weak-to-strong generalization problem and solutions. Discuss superalignment challenges. Include interpretability research and safety evaluation methods. Reference Anthropic's specific contributions to field.

---

### **Week 12: Reward Hacking & Safety**
**Study Topics:**
- Goodhart's law in AI
- Distributional shift in RLHF
- Overoptimization problems
- **Core Papers**: Reward hacking papers, safety research

**Experiments:**
- Demonstrate reward hacking in simple environments
- Study overoptimization effects
- **WandB**: Safety metrics, robustness analysis

**Blog**: "When Optimization Goes Wrong: The Dark Side of Reward Maximization"
**Content**: Reference all previous RLHF work. Show concrete examples of reward hacking. Discuss safety implications and mitigation strategies.

---

## **Phase 4: Advanced Topics (Weeks 13-16)**

### **Week 13: Scaling Laws & Emergent Capabilities**
**Study Topics:**
- **Foundational Scaling Laws**:
  - Original GPT-3 scaling laws (Kaplan et al.)
  - Chinchilla scaling laws (compute-optimal training)
  - PaLM scaling analysis and insights
- **Advanced Scaling Research**:
  - **Inverse scaling**: Cases where bigger models perform worse
  - **Beyond chinchilla**: Latest optimal training research
  - **Scaling for different capabilities**: Reasoning, coding, math
  - **Transfer scaling**: How pre-training transfers to downstream tasks
- **Emergent Capabilities Research**:
  - **Capability emergence**: Chain-of-thought, in-context learning thresholds
  - **Grokking phenomena**: Sudden emergence during training
  - **Few-shot emergence**: How capabilities emerge with scale
  - **Mechanistic interpretability**: Understanding emergence through circuits
- **Latest Scaling Insights**:
  - **Data quality vs quantity**: Scaling with high-quality data
  - **Multimodal scaling**: Vision-language model scaling patterns
  - **Post-training scaling**: RLHF and instruction tuning scaling
  - **Inference-time scaling**: Test-time compute scaling laws
- **Practical Scaling Considerations**:
  - **Resource allocation**: Compute vs data vs parameter trade-offs
  - **Scaling infrastructure**: Distributed training considerations
  - **Scaling evaluation**: How to measure capabilities at scale
- **Core Papers**: 
  - "Scaling Laws for Neural Language Models" (Kaplan et al.)
  - "Training Compute-Optimal Large Language Models" (Chinchilla)
  - "Emergent Abilities of Large Language Models" (Wei et al.)
  - "Inverse Scaling: When Bigger Isn't Better"

**Experiments:**
- Validate scaling laws on controlled model family
- Emergence detection experiments across model sizes
- Data quality vs quantity scaling studies
- Capability evaluation across scales
- **WandB**: Scaling curves, emergence detection, capability thresholds

**Blog 1**: "The Mathematics of Intelligence: Decoding Scaling Laws"
**Notes for Agent**: Start with foundational scaling laws, build through Chinchilla to latest research. Show empirical relationships and theoretical understanding. Include experimental validation and practical implications for model development.

**Blog 2**: "When Magic Happens: Understanding Emergent Capabilities in Large Models"
**Notes for Agent**: Deep dive into emergence phenomena. Explain chain-of-thought emergence, grokking, few-shot learning emergence. Include experimental detection methods and theoretical frameworks. Connect to mechanistic interpretability research.

---

### **Week 14: Value-Based Methods**
**Study Topics:**
- Deep Q-Networks (DQN)
- Experience replay
- Double DQN and improvements
- **Core Papers**: DQN paper, Rainbow paper

**Experiments:**
- Implement DQN with experience replay
- Compare with policy methods
- **WandB**: Sample efficiency analysis

**Blog**: "Learning Values: From Bellman to Deep Q-Networks"
**Content**: Reference value functions from Week 1. Show evolution to deep networks. Compare with policy methods from earlier weeks.

---

### **Week 15: Model-Based RL**
**Study Topics:**
- World models and planning
- Model-based vs model-free trade-offs
- Dyna-Q concepts
- **Core Papers**: Model-based RL survey, world models paper

**Experiments:**
- Simple model-based RL implementation
- Planning vs learning comparison
- **WandB**: Sample efficiency, planning accuracy

**Blog**: "Planning vs Learning: The Model-Based Advantage"
**Content**: Contrast with all previous model-free methods. Show benefits of world models. Connect to transformer's implicit modeling.

---

### **Week 16: Research Frontiers**
**Study Topics:**
- Current open problems
- Meta-learning and few-shot adaptation
- Multi-agent considerations
- **Core Papers**: Recent research directions

**Experiments:**
- Implement simple meta-learning
- Survey current techniques
- **WandB**: Meta-learning curves

**Blog**: "The Road Ahead: Open Problems and Future Directions"
**Content**: Synthesize all previous work. Identify gaps and opportunities. Outline personal research interests.

---

## **Key Improvements with Agent-Assisted Blogging:**

**✅ Accelerated Publishing:**
- **2 blogs per week** consistently achievable
- **30-45 minutes** for blog review vs 2-3 hours writing
- **More time for deep learning** and experimentation

**✅ Enhanced Content Quality:**
- **Detailed note-taking** feeds better blog generation
- **Focus on technical accuracy** rather than writing mechanics
- **Consistent publishing rhythm** builds stronger online presence

**✅ Expanded Scope:**
- **Primary concept + connection/experiment post** each week
- **Cross-topic synthesis** posts become easier
- **Real-time learning documentation** through better notes

**✅ Optimized Learning Loop:**
- **More experimental iterations** per topic
- **Deeper WandB experiment tracking**
- **Time for advanced implementations**

**Total Time Commitment**: ~5-6 hours per week (4-5 study/code, 30-45 minutes blog review)
**Blog Output**: 40 high-quality posts over 20 weeks (2 per week)
**Course Coverage**: 100% of CS336 and CS285 core topics + cutting-edge research
**Anthropic Relevance**: Directly targets their research priorities + latest developments

---

## **🚀 Cutting-Edge Topics Beyond Standard Tutorials:**

**Advanced Attention & Architecture:**
- Flash Attention, Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
- RoPE, ALiBi, complex rotary embeddings
- Mamba, RetNet, State Space Models
- MoE (Mixture of Experts), sparse attention patterns
- LLaMA, PaLM architectural innovations

**Latest Policy Optimization:**
- GRPO (Group Relative Policy Optimization)
- DPO, IPO, SimPO, KTO (latest preference optimization)
- Constitutional AI methods
- RLAIF and self-rewarding models

**Advanced Training & Optimization:**
- Lion optimizer, Adafactor variants
- ZeRO memory optimization
- Mixed precision (FP16, BF16, FP8)
- Distributed training optimization techniques

**Safety & Alignment Research:**
- Weak-to-strong generalization
- Superalignment research
- Advanced interpretability (sparse autoencoders)
- Process vs outcome supervision
- Scalable oversight methods

**Multimodal & Tool Use:**
- GPT-4V, LLaVA multimodal architectures
- CLIP, DALL-E, Flamingo innovations
- ReAct, Toolformer tool-using frameworks
- RAG (Retrieval-Augmented Generation) advances

**Meta-Learning & Emergent Capabilities:**
- In-context learning mechanisms
- Chain-of-thought emergence analysis
- MAML and modern meta-learning
- Scaling law frontiers and inverse scaling

**Research Frontiers:**
- Neurosymbolic AI approaches
- Causal reasoning in AI
- AI for scientific discovery
- Quantum ML intersections

---

## **🎯 Anthropic-Specific Advantages:**

**Constitutional AI Mastery**: Deep coverage of Anthropic's key innovation
**Safety-First Approach**: Every topic includes safety considerations
**Preference Learning Expertise**: Complete coverage from basic to cutting-edge methods
**Interpretability Focus**: Understanding model internals and behavior
**Scaling Research**: Latest insights on model scaling and emergence
**Practical Implementation**: All concepts backed by working code

This roadmap ensures you're not just learning the basics but mastering the cutting-edge research that Anthropic values most. You'll be discussing concepts that most ML practitioners haven't even heard of!