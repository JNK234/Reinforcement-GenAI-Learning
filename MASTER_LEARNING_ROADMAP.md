# RL & GenAI Master Learning Roadmap
*Your comprehensive learning journey - Living Document*

## Progress Legend
- ✓ **Completed** - Fully understood and notes created
- 🔄 **In Progress** - Currently studying
- ⏳ **Planned** - Scheduled for future
- 🌟 **Blog Published** - Content shared publicly

## Overall Progress Dashboard
```
Clusters Completed: 0/24 | Blogs Published: 0/24 | Current Focus: RL Foundations
```

---

## **PHASE 1: RL FOUNDATIONS**

### Cluster 1.1: Sequential Decision Making ⏳
**Blog Scope**: "How AI Makes Decisions Step by Step" (12-15 min read)
**Topics**: Markov processes, sequential decisions, states/actions/rewards
**Resources**:
- **Primary**: [CS234 Lecture 1](https://web.stanford.edu/class/cs234/slides/cs234_lecture1.pdf) - Core framework
- **Primary**: [Sutton & Barto Ch 1](http://incompleteideas.net/book/RLbook2020.pdf#page=17) - Mathematical foundations
- **Supplementary**: [OpenAI Spinning Up - RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) - Practical perspective
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 1.2: Markov Decision Processes ⏳
**Blog Scope**: "The Mathematical Foundation of AI Decision Making" (12-15 min read)
**Topics**: MDP formalism, states, actions, rewards, transitions, policies
**Resources**:
- **Primary**: [CS234 Lecture 2](https://web.stanford.edu/class/cs234/slides/cs234_lecture2.pdf) - Complete MDP theory
- **Primary**: [Sutton & Barto Ch 3](http://incompleteideas.net/book/RLbook2020.pdf#page=65) - Mathematical rigor
- **Supplementary**: [David Silver Lecture 2](https://www.youtube.com/watch?v=lfHX2hHRMVQ) - Alternative explanations
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 1.3: Bellman Equations ⏳
**Blog Scope**: "Finding the Best Path: Bellman's Insight" (12-15 min read)
**Topics**: Value functions, Bellman optimality, recursive structure
**Resources**:
- **Primary**: [CS234 Lecture 2 (Bellman section)](https://web.stanford.edu/class/cs234/slides/cs234_lecture2.pdf) - Derivations
- **Primary**: [Sutton & Barto Ch 3.5-3.8](http://incompleteideas.net/book/RLbook2020.pdf#page=75) - Mathematical depth
- **Supplementary**: [David Silver Slides - Bellman](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf) - Visual approach
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 1.4: Dynamic Programming ⏳
**Blog Scope**: "Perfect Planning in a Known World" (12-15 min read)
**Topics**: Value iteration, policy iteration, convergence guarantees
**Resources**:
- **Primary**: [CS234 Lecture 3](https://web.stanford.edu/class/cs234/slides/cs234_lecture3.pdf) - Algorithm details
- **Primary**: [Sutton & Barto Ch 4](http://incompleteideas.net/book/RLbook2020.pdf#page=89) - Theoretical foundations
- **Supplementary**: [Bertsekas DP Ch 1](http://www.athenasc.com/dpchapter.pdf) - Advanced theory
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## **PHASE 2: MODEL-FREE METHODS**

### Cluster 2.1: Monte Carlo Methods ⏳
**Blog Scope**: "Learning from Complete Episodes" (12-15 min read)
**Topics**: First-visit MC, every-visit MC, exploration vs exploitation
**Resources**:
- **Primary**: [Sutton & Barto Ch 5](http://incompleteideas.net/book/RLbook2020.pdf#page=111) - Core theory
- **Primary**: [CS234 Lecture 4](https://web.stanford.edu/class/cs234/slides/cs234_lecture4.pdf) - Practical aspects
- **Supplementary**: [Monte Carlo Tutorial](https://github.com/dennybritz/reinforcement-learning/tree/master/MC) - Implementation
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 2.2: Temporal Difference Learning ⏳
**Blog Scope**: "Learning from Every Step" (12-15 min read)
**Topics**: TD(0), TD vs MC comparison, bootstrapping
**Resources**:
- **Primary**: [Sutton & Barto Ch 6.1-6.3](http://incompleteideas.net/book/RLbook2020.pdf#page=129) - Foundational theory
- **Primary**: [David Silver TD Learning](https://www.youtube.com/watch?v=0g4j2k_Ggc4) - Clear explanations
- **Supplementary**: [CS285 TD Slides](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) - Advanced perspective
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 2.3: Q-Learning Fundamentals ⏳
**Blog Scope**: "Off-Policy Learning with Q-Functions" (12-15 min read)
**Topics**: Q-learning algorithm, off-policy learning, convergence
**Resources**:
- **Primary**: [Original Q-Learning Paper](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) - Historical foundation
- **Primary**: [CS234 Lecture 5](https://web.stanford.edu/class/cs234/slides/cs234_lecture5.pdf) - Modern perspective
- **Supplementary**: [Q-Learning Tutorial](https://github.com/dennybritz/reinforcement-learning/tree/master/TD) - Implementation
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 2.4: SARSA and On-Policy Methods ⏳
**Blog Scope**: "On-Policy vs Off-Policy: When It Matters" (12-15 min read)
**Topics**: SARSA, Expected SARSA, on-policy vs off-policy comparison
**Resources**:
- **Primary**: [Sutton & Barto Ch 6.4-6.6](http://incompleteideas.net/book/RLbook2020.pdf#page=145) - Theoretical differences
- **Primary**: [CS234 SARSA Materials](https://web.stanford.edu/class/cs234/slides/cs234_lecture5.pdf) - Practical comparison
- **Supplementary**: [TD Methods Comparison](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdmethods.html) - Visual guide
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## **PHASE 3: DEEP RL FOUNDATIONS**

### Cluster 3.1: Function Approximation ⏳
**Blog Scope**: "From Tables to Neural Networks in RL" (12-15 min read)
**Topics**: Why function approximation, linear methods, neural network basics
**Resources**:
- **Primary**: [Sutton & Barto Ch 9](http://incompleteideas.net/book/RLbook2020.pdf#page=217) - Theoretical foundation
- **Primary**: [CS285 Function Approximation](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf) - Deep learning perspective
- **Supplementary**: [PyTorch Deep Learning Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) - Implementation basics
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 3.2: Deep Q-Networks (DQN) ⏳
**Blog Scope**: "Breaking the Atari Barrier: How DQN Works" (12-15 min read)
**Topics**: DQN architecture, experience replay, target networks, stability tricks
**Resources**:
- **Primary**: [Nature DQN Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - Original breakthrough
- **Primary**: [CS285 DQN Lecture](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf) - Technical details
- **Supplementary**: [CleanRL DQN Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py) - Clean code
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 3.3: DQN Improvements ⏳
**Blog Scope**: "Evolution of Deep Q-Learning: Double, Dueling, and Beyond" (12-15 min read)
**Topics**: Double DQN, Dueling DQN, Prioritized Experience Replay, Rainbow
**Resources**:
- **Primary**: [Rainbow DQN Paper](https://arxiv.org/pdf/1710.02298.pdf) - Comprehensive improvements
- **Primary**: [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) - Key innovation
- **Supplementary**: [DQN Improvements Survey](https://arxiv.org/pdf/1710.02298.pdf) - Complete overview
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## **PHASE 4: POLICY GRADIENT METHODS**

### Cluster 4.1: Policy Gradient Theory ⏳
**Blog Scope**: "From Values to Policies: Direct Optimization" (12-15 min read)
**Topics**: Policy gradient theorem, REINFORCE, baseline methods
**Resources**:
- **Primary**: [Policy Gradient Theorem Paper](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) - Mathematical foundation
- **Primary**: [CS285 Policy Gradients](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf) - Modern perspective
- **Supplementary**: [Lil'Log Policy Gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) - Accessible explanation
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 4.2: Actor-Critic Methods ⏳
**Blog Scope**: "Best of Both Worlds: Actor-Critic Architecture" (12-15 min read)
**Topics**: Basic actor-critic, A2C, advantage estimation
**Resources**:
- **Primary**: [CS285 Actor-Critic Slides](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf) - Technical details
- **Primary**: [A2C Paper](https://arxiv.org/pdf/1602.01783.pdf) - Practical implementation
- **Supplementary**: [Actor-Critic Tutorial](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient) - Code walkthrough
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 4.3: Trust Region Methods ⏳
**Blog Scope**: "Stable Policy Updates: TRPO and Beyond" (12-15 min read)
**Topics**: Natural policy gradients, TRPO, trust region constraints
**Resources**:
- **Primary**: [TRPO Paper](https://arxiv.org/pdf/1502.05477.pdf) - Original method
- **Primary**: [Natural Policy Gradients](https://papers.nips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf) - Theoretical foundation
- **Supplementary**: [Trust Region Tutorial](https://spinningup.openai.com/en/latest/algorithms/trpo.html) - Practical guide
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 4.4: Proximal Policy Optimization (PPO) ⏳
**Blog Scope**: "PPO: The Workhorse of Modern RL" (12-15 min read)
**Topics**: PPO algorithm, clipped objective, implementation details
**Resources**:
- **Primary**: [PPO Paper](https://arxiv.org/pdf/1707.06347.pdf) - Original algorithm
- **Primary**: [37 PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Practical insights
- **Supplementary**: [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) - Reference implementation
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## **PHASE 5: TRANSFORMER FOUNDATIONS**

### Cluster 5.1: Attention Mechanisms ⏳
**Blog Scope**: "The Attention Revolution: Information Routing in AI" (12-15 min read)
**Topics**: Scaled dot-product attention, multi-head attention, self-attention
**Resources**:
- **Primary**: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Revolutionary paper
- **Primary**: [CS336 Attention Lecture](https://stanford-cs336.github.io/spring2025/) - Modern perspective
- **Supplementary**: [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 5.2: Transformer Architecture ⏳
**Blog Scope**: "Building Blocks of Language Models: Transformer Deep Dive" (12-15 min read)
**Topics**: Layer normalization, positional encoding, feed-forward networks
**Resources**:
- **Primary**: [Transformer Paper](https://arxiv.org/pdf/1706.03762.pdf) - Complete architecture
- **Primary**: [CS336 Transformer Materials](https://stanford-cs336.github.io/spring2025/) - Implementation details
- **Supplementary**: [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal implementation
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 5.3: Language Modeling ⏳
**Blog Scope**: "Teaching Machines to Speak: Autoregressive Generation" (12-15 min read)
**Topics**: Autoregressive generation, cross-entropy loss, perplexity, scaling
**Resources**:
- **Primary**: [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - Foundation
- **Primary**: [CS336 Language Modeling](https://stanford-cs336.github.io/spring2025/) - Modern approach
- **Supplementary**: [Karpathy Neural Networks Course](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - From scratch
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 5.4: Tokenization Deep Dive ⏳
**Blog Scope**: "Breaking Down Language: How AI Reads Text" (12-15 min read)
**Topics**: BPE, WordPiece, SentencePiece, vocabulary construction
**Resources**:
- **Primary**: [BPE Paper](https://arxiv.org/pdf/1508.07909.pdf) - Foundational method
- **Primary**: [SentencePiece Paper](https://arxiv.org/pdf/1808.06226.pdf) - Modern approach
- **Supplementary**: [Tokenization Tutorial](https://huggingface.co/learn/nlp-course/chapter6/1) - Practical implementation
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## **PHASE 6: RLHF & ALIGNMENT**

### Cluster 6.1: Human Preference Learning ⏳
**Blog Scope**: "Teaching AI Human Values: Preference Learning" (12-15 min read)
**Topics**: Bradley-Terry models, reward modeling, preference datasets
**Resources**:
- **Primary**: [InstructGPT Paper](https://arxiv.org/pdf/2203.02155.pdf) - RLHF foundation
- **Primary**: [Constitutional AI](https://arxiv.org/pdf/2212.08073.pdf) - Advanced methods
- **Supplementary**: [RLHF Blog Post](https://huggingface.co/blog/rlhf) - Practical overview
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 6.2: PPO for Language Models ⏳
**Blog Scope**: "RLHF in Practice: Optimizing Language Models" (12-15 min read)
**Topics**: PPO adaptation for text, KL penalties, sequence-level rewards
**Resources**:
- **Primary**: [InstructGPT Technical Details](https://arxiv.org/pdf/2203.02155.pdf) - Implementation
- **Primary**: [TRL Library](https://github.com/huggingface/trl) - Practical tools
- **Supplementary**: [RLHF Tutorial](https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-Human-Feedback--VmlldzozMzAwOTB1) - Step-by-step
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 6.3: Constitutional AI & Safety ⏳
**Blog Scope**: "Building Safe AI: Constitutional Methods" (12-15 min read)
**Topics**: Constitutional training, AI feedback, safety considerations
**Resources**:
- **Primary**: [Constitutional AI Paper](https://arxiv.org/pdf/2212.08073.pdf) - Complete method
- **Primary**: [Anthropic Safety Research](https://www.anthropic.com/research) - Latest developments
- **Supplementary**: [AI Safety Blog](https://www.safe.ai/) - Broader context
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 6.4: Advanced Preference Methods ⏳
**Blog Scope**: "Beyond RLHF: DPO, IPO, and the Future" (12-15 min read)
**Topics**: DPO, IPO, SimPO, KTO, reward-free methods
**Resources**:
- **Primary**: [DPO Paper](https://arxiv.org/pdf/2305.18290.pdf) - Breakthrough method
- **Primary**: [IPO Paper](https://arxiv.org/pdf/2310.12036.pdf) - Alternative approach
- **Supplementary**: [Preference Optimization Survey](https://arxiv.org/pdf/2401.01045.pdf) - Complete overview
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## **PHASE 7: ADVANCED TOPICS**

### Cluster 7.1: Scaling Laws ⏳
**Blog Scope**: "The Mathematics of Intelligence: Understanding Scaling" (12-15 min read)
**Topics**: Chinchilla laws, emergent abilities, compute-optimal training
**Resources**:
- **Primary**: [Scaling Laws Paper](https://arxiv.org/pdf/2001.08361.pdf) - Original research
- **Primary**: [Chinchilla Paper](https://arxiv.org/pdf/2203.15556.pdf) - Compute-optimal scaling
- **Supplementary**: [Emergent Abilities](https://arxiv.org/pdf/2206.07682.pdf) - Capability emergence
**Status**: ⏳ **Learning Time**: 2-3 hours

### Cluster 7.2: Advanced Architectures ⏳
**Blog Scope**: "Beyond Transformers: LLaMA, Mamba, and the Future" (12-15 min read)
**Topics**: LLaMA innovations, Mamba/SSMs, MoE architectures
**Resources**:
- **Primary**: [LLaMA Paper](https://arxiv.org/pdf/2302.13971.pdf) - Architectural innovations
- **Primary**: [Mamba Paper](https://arxiv.org/pdf/2312.00752.pdf) - Alternative to attention
- **Supplementary**: [Switch Transformer](https://arxiv.org/pdf/2101.03961.pdf) - MoE approach
**Status**: ⏳ **Learning Time**: 2-3 hours

---

## Resource Addition Template

When you find new resources, add them using this format:

```markdown
### New Resource: [TITLE]
**Type**: [course/paper/video/book/blog]
**URL**: [link]
**Topics Covered**: [list]
**Suggested Cluster**: [X.Y] or [New Cluster]
**Priority**: [Primary/Supplementary/Optional]
**Integration Notes**: [how it fits with existing materials]
```

---

## Progress Tracking Notes

- **Completion Criteria**: Can explain concept clearly + implement if applicable + ready to blog
- **Review Schedule**: Monthly review of completed clusters for retention
- **Blog Pipeline**: Each completed cluster → blog post within 1 week
- **Knowledge Gaps**: Track concepts that need revisiting