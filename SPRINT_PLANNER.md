# Learning Sprint Planner
*Sequential Track Learning | Target: 2 Blogs per Week*

## Current Sprint: **Sprint 2 - Sequential Track Learning to RLHF Convergence**

**Dates**: August 5-30, 2025 (25 days)  
**Focus**: **RL Foundations → GenAI Foundations → RLHF Convergence**  
**Goal**: Master RL decision-making, then language model architecture, culminating in RLHF mastery  
**Status**: Starting - Sequential Decision Making Completed ✅

**Strategic Approach**: Build deep foundations in each domain separately, then bring them together where they naturally converge in RLHF.

**Why This Works**:

- **Week 1-2**: RL mastery (MDP, Bellman, Q-Learning, Policy Gradients)
- **Week 2-3**: GenAI mastery (Attention, Transformers, Language Modeling)  
- **Week 3-4**: Natural convergence (RLHF, Constitutional AI, DPO)
- Deep focus prevents cognitive overload and ensures retention

---

## Sequential Focus Structure

**Dedicated 2.5-hour sessions per domain:**

- **Sessions 1-4**: Pure RL deep dives (mathematical foundations + implementation)
- **Sessions 5-8**: Pure GenAI deep dives (architecture + language modeling)
- **Sessions 9-12**: Convergence sessions (RLHF + modern preference methods)

**Learning Philosophy**: Master each domain completely before integration, then see how they naturally unite in cutting-edge AI systems.

---

## **TRACK A: RL FOUNDATIONS (Sessions 1-4)**

### **Session 1: Monday, August 5** (2.5 hours)
**🎯 Target**: Cluster 1.2 - Markov Decision Processes (MDP)  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [CS234 Lecture 2](https://web.stanford.edu/class/cs234/slides/cs234_lecture2.pdf) - Complete MDP theory
- ⭐ [Sutton & Barto Chapter 3](http://incompleteideas.net/book/RLbook2020.pdf#page=65) - Mathematical rigor (Pages 47-88)
- 🎥 [David Silver Lecture 2](https://www.youtube.com/watch?v=lfHX2hHRMVQ&list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb) - MDP Formalism (1h 37m)
- 🎯 [Interactive MDP Examples](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html) - Gridworld visualization
- 📚 [Berkeley CS188 MDP Notes](https://inst.eecs.berkeley.edu/~cs188/sp20/assets/notes/n6.pdf) - Alternative perspective

**⏰ Time Allocation**:

- 30 min: Connect to your blog's sequential decision concepts
- 90 min: Deep dive into MDP formalism (states, actions, rewards, transitions)
- 30 min: Work through concrete MDP examples
- 30 min: Practice formalizing problems as MDPs

**🎯 Success Criteria**:

- Can formalize any decision problem as an MDP
- Understand mathematical notation (S, A, R, T, γ)
- Distinguish finite vs infinite horizon problems
- Ready to write "The Mathematical Foundation of AI Decision Making"

**🔧 Knowledge Forge Focus**:

- How does MDP formalism capture real-world problems?
- What makes the Markov property so powerful?
- When do MDP assumptions break down in practice?

---

### **Session 2: Thursday, August 8** (2.5 hours)
**🎯 Target**: Cluster 1.3 - Bellman Equations  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [CS234 Lecture 2 - Bellman Section](https://web.stanford.edu/class/cs234/slides/cs234_lecture2.pdf) - Derivations
- ⭐ [Sutton & Barto Chapter 3.5-3.8](http://incompleteideas.net/book/RLbook2020.pdf#page=75) - Mathematical depth (Pages 58-75)
- 🎥 [David Silver Lecture 2 - Bellman](https://www.youtube.com/watch?v=lfHX2hHRMVQ&t=2400s) - Timestamped to Bellman section
- 🎯 [Interactive Bellman Demo](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html) - See equations in action
- 📚 [MIT 6.034 Bellman Notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lectures/) - Clear derivations

**⏰ Time Allocation**:

- 30 min: Review MDP concepts and connect to value functions
- 90 min: Derive Bellman equations from first principles
- 30 min: Understand recursive structure of optimal solutions
- 30 min: Work through examples and visualizations

**🎯 Success Criteria**:

- Can derive Bellman optimality equations
- Understand recursive nature of optimal value functions
- Grasp relationship between V* and Q*
- Ready to write "Finding the Best Path: Bellman's Insight"

**🔧 Mathematical Focus**:

- V*(s) = max_a Σ p(s'|s,a)[r + γV*(s')]
- Q*(s,a) = Σ p(s'|s,a)[r + γ max_a' Q*(s',a')]
- Why these equations guarantee optimality

---

### **Session 3: Monday, August 12** (2.5 hours)
**🎯 Target**: Cluster 1.4 - Dynamic Programming + Cluster 2.3 - Q-Learning Implementation  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [CS234 Lecture 3](https://web.stanford.edu/class/cs234/slides/cs234_lecture3.pdf) - Algorithm details
- ⭐ [Sutton & Barto Chapter 4](http://incompleteideas.net/book/RLbook2020.pdf#page=89) - Theoretical foundations (Pages 75-108)
- ⭐ [Original Q-Learning Paper](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) - Historical foundation
- 🛠️ [HuggingFace Deep RL Units 1-2](https://huggingface.co/learn/deep-rl-course/en/unit1/introduction) - Q-Learning implementation
- 🎯 [OpenAI Gym Tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/) - Environment setup

**⏰ Time Allocation**:

- 30 min: Connect Bellman equations to iterative algorithms
- 60 min: Implement value iteration and policy iteration
- 60 min: Implement Q-learning from scratch for FrozenLake
- 30 min: Compare model-based vs model-free approaches

**🎯 Success Criteria**:

- Working value iteration and policy iteration
- Working Q-learning implementation solving FrozenLake
- Understand when DP works vs when Q-learning is needed
- Ready to write "From Perfect Planning to Learning Through Trial"

---

### **Session 4: Thursday, August 15** (2.5 hours)
**🎯 Target**: Cluster 2.2 - Temporal Difference Learning + Cluster 4.4 - PPO Introduction  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [Sutton & Barto Chapter 6.1-6.3](http://incompleteideas.net/book/RLbook2020.pdf#page=129) - TD Learning foundations
- ⭐ [David Silver TD Learning](https://www.youtube.com/watch?v=0g4j2k_Ggc4) - Clear explanations
- ⭐ [PPO Paper](https://arxiv.org/pdf/1707.06347.pdf) - Original algorithm
- 🛠️ [37 PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Practical insights
- 🛠️ [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) - Reference implementation

**⏰ Time Allocation**:

- 30 min: Compare TD vs Monte Carlo methods
- 60 min: Implement SARSA and compare with Q-Learning
- 60 min: Understand PPO algorithm and policy gradient basics
- 30 min: Connect value-based to policy-based methods

**🎯 Success Criteria**:

- Understand bootstrapping vs full episodes
- Working SARSA implementation
- Conceptual understanding of PPO for future RLHF
- Ready to write "Learning from Every Step: The TD Revolution"

---

## **TRACK B: GENAI FOUNDATIONS (Sessions 5-8)**

### **Session 5: Monday, August 19** (2.5 hours)
**🎯 Target**: Cluster 5.1 - Attention Mechanisms (Foundational)  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Revolutionary paper (Focus: Sections 1-3.2)
- ⭐ [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide to attention
- 🎥 [3Blue1Brown - Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) - Visual intuition (13 min)
- 📄 [CS336 Attention Lecture](https://stanford-cs336.github.io/spring2025/) - Modern perspective
- 🛠️ [Attention from Scratch Tutorial](https://peterbloem.nl/blog/transformers) - Implementation basics

**⏰ Time Allocation**:

- 30 min: Why attention was revolutionary (sequence modeling problems)
- 90 min: Deep dive into scaled dot-product attention (Q, K, V mechanics)
- 60 min: Multi-head attention and self-attention
- 30 min: Implement basic attention mechanism

**🎯 Success Criteria**:

- Understand Query, Key, Value concept deeply
- Can implement scaled dot-product attention
- Grasp why attention solved sequence modeling problems
- Ready to write "The Attention Revolution: How AI Learned to Focus"

---

### **Session 6: Thursday, August 22** (2.5 hours)
**🎯 Target**: Cluster 5.2 - Transformer Architecture (Complete)  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Complete architecture (Sections 3.3-4)
- ⭐ [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Karpathy implementation (1h 57m)
- 🛠️ [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal implementation reference
- 📄 [CS336 Transformer Materials](https://stanford-cs336.github.io/spring2025/) - Implementation details
- 🎯 [Transformer Math Walkthrough](https://nlp.seas.harvard.edu/annotated-transformer/) - Annotated implementation

**⏰ Time Allocation**:

- 30 min: Connect attention to full transformer architecture
- 90 min: Layer normalization, positional encoding, feed-forward networks
- 60 min: Build minimal transformer from scratch
- 30 min: Understand training dynamics and scaling

**🎯 Success Criteria**:

- Complete understanding of transformer architecture
- Working minimal transformer implementation
- Understand each component's role (norm, pos, FFN)
- Ready to write "Building Blocks of Language Models: Transformer Deep Dive"

---

### **Session 7: Monday, August 26** (2.5 hours)
**🎯 Target**: Cluster 5.3 - Language Modeling + Cluster 5.4 - Tokenization  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - Foundation
- ⭐ [CS336 Language Modeling](https://stanford-cs336.github.io/spring2025/) - Modern approach
- ⭐ [BPE Paper](https://arxiv.org/pdf/1508.07909.pdf) - Foundational tokenization method
- ⭐ [SentencePiece Paper](https://arxiv.org/pdf/1808.06226.pdf) - Modern approach
- 🛠️ [Tokenization Tutorial](https://huggingface.co/learn/nlp-course/chapter6/1) - Practical implementation
- 🎥 [Karpathy Neural Networks Course](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - From scratch approach

**⏰ Time Allocation**:

- 30 min: Connect transformers to language modeling task
- 60 min: Autoregressive generation, cross-entropy loss, perplexity
- 60 min: Tokenization deep dive (BPE, WordPiece, SentencePiece)
- 30 min: Train small language model on simple dataset

**🎯 Success Criteria**:

- Understand autoregressive generation deeply
- Can implement and train language model
- Master tokenization approaches and trade-offs
- Ready to write "Teaching Machines to Speak: Language Modeling Fundamentals"

---

### **Session 8: Thursday, August 29** (2.5 hours)
**🎯 Target**: Advanced Architectures + Scaling Preparation for RLHF  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [Scaling Laws Paper](https://arxiv.org/pdf/2001.08361.pdf) - Original research
- ⭐ [Chinchilla Paper](https://arxiv.org/pdf/2203.15556.pdf) - Compute-optimal scaling
- ⭐ [LLaMA Paper](https://arxiv.org/pdf/2302.13971.pdf) - Architectural innovations
- 📄 [Emergent Abilities](https://arxiv.org/pdf/2206.07682.pdf) - Capability emergence
- 🛠️ [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) - Practical model usage
- 📚 [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) - Karpathy overview

**⏰ Time Allocation**:

- 30 min: Review transformer and language modeling foundations
- 60 min: Scaling laws and emergent abilities
- 60 min: Modern architectural improvements (LLaMA, RMSNorm, etc.)
- 30 min: Prepare for RLHF - understanding base model training

**🎯 Success Criteria**:

- Understand scaling laws and their implications
- Familiar with modern LLM architectures
- Ready for RLHF - solid base model understanding
- Ready to write "The Mathematics of Intelligence: Understanding Scaling"

---

## **TRACK C: RLHF CONVERGENCE (Sessions 9-12)**

### **Session 9: Monday, September 2** (2.5 hours)
**🎯 Target**: Cluster 6.1 - Human Preference Learning (Foundation)  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [InstructGPT Paper](https://arxiv.org/pdf/2203.02155.pdf) - RLHF foundation
- ⭐ [RLHF Blog Post](https://huggingface.co/blog/rlhf) - Practical overview
- 📄 [CMU RLHF 101 Tutorial](https://sites.google.com/andrew.cmu.edu/rlhf-tutorial/home) - 2025 comprehensive guide
- 🎥 [Anthropic RLHF Explainer](https://www.anthropic.com/research/rlhf) - Constitutional AI perspective
- 🛠️ [TRL Library](https://github.com/huggingface/trl) - Practical RLHF tools

**⏰ Time Allocation**:

- 30 min: Connect RL foundations to language model training
- 90 min: Bradley-Terry models, reward modeling, preference datasets
- 60 min: RLHF pipeline - supervised fine-tuning → reward modeling → PPO
- 30 min: Implement basic preference comparison

**🎯 Success Criteria**:

- Understand complete RLHF pipeline
- See how RL concepts apply to language models
- Can implement basic reward model
- Ready to write "Teaching AI Human Values: The RLHF Revolution"

---

### **Session 10: Thursday, September 5** (2.5 hours)
**🎯 Target**: Cluster 6.2 - PPO for Language Models  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [InstructGPT Technical Details](https://arxiv.org/pdf/2203.02155.pdf) - Implementation specifics
- ⭐ [TRL Library](https://github.com/huggingface/trl) - Practical tools
- 🛠️ [RLHF Tutorial](https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-Human-Feedback--VmlldzozMzAwOTB1) - Step-by-step
- 📄 [37 PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) - Critical insights
- 🎯 [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) - Open-source implementation

**⏰ Time Allocation**:

- 30 min: Review PPO fundamentals from Session 4
- 90 min: PPO adaptation for text generation (KL penalties, sequence rewards)
- 60 min: Implement RLHF training loop
- 30 min: Debug common RLHF training issues

**🎯 Success Criteria**:

- Working RLHF implementation
- Understand KL penalty and why it's needed
- Can train language model with human feedback
- Ready to write "RLHF in Practice: Training Language Models with PPO"

---

### **Session 11: Monday, September 9** (2.5 hours)
**🎯 Target**: Cluster 6.3 - Constitutional AI + Cluster 6.4 - Advanced Preference Methods  
**📚 Materials to Upload to NotebookLM**:

- ⭐ [Constitutional AI Paper](https://arxiv.org/pdf/2212.08073.pdf) - Complete method
- ⭐ [DPO Paper](https://arxiv.org/pdf/2305.18290.pdf) - Breakthrough method
- ⭐ [SimPO Paper](https://arxiv.org/abs/2405.14734) - Princeton 2025, latest advancement
- 📄 [IPO Paper](https://arxiv.org/pdf/2310.12036.pdf) - Alternative approach
- 📚 [Preference Optimization Survey](https://arxiv.org/pdf/2401.01045.pdf) - Complete overview
- 🛠️ [Anthropic Safety Research](https://www.anthropic.com/research) - Latest developments

**⏰ Time Allocation**:

- 30 min: Limitations of RLHF and need for improvements
- 60 min: Constitutional AI - AI feedback vs human feedback
- 60 min: DPO, IPO, SimPO - reward-free preference optimization
- 30 min: Compare all preference optimization methods

**🎯 Success Criteria**:

- Understand Constitutional AI approach
- Can implement DPO as alternative to RLHF
- Know cutting-edge preference optimization methods
- Ready to write "Beyond RLHF: DPO, Constitutional AI, and the Future"

---

### **Session 12: Thursday, September 12** (2.5 hours)
**🎯 Target**: Integration Mastery + Future Directions  
**📚 Materials to Upload to NotebookLM**:

- 📄 Review all previous session materials
- ⭐ [LLaMA 2 Paper](https://arxiv.org/pdf/2307.09288.pdf) - Modern RLHF at scale
- 📚 [Anthropic Constitutional AI](https://arxiv.org/pdf/2212.08073.pdf) - Advanced safety
- 🔮 [AI Alignment Research](https://www.safe.ai/) - Future directions
- 🛠️ [Complete RLHF Implementation](https://github.com/CarperAI/trlx) - Production-ready tools

**⏰ Time Allocation**:

- 45 min: Synthesize complete learning journey (RL → GenAI → RLHF)
- 45 min: Implement end-to-end RLHF pipeline
- 45 min: Explore cutting-edge research and future directions
- 45 min: Plan next learning phase and specialization areas

**🎯 Success Criteria**:

- Complete mastery of RL-to-RLHF pipeline
- Working knowledge of all major preference optimization methods
- Clear understanding of current research frontiers
- Ready to write "RLHF Mastery: From Bellman Equations to Constitutional AI"

---

## Expected Blog Outputs

### **RL Track Blogs (Sessions 1-4)**

1. **"The Mathematical Foundation of AI Decision Making"** (Session 1 - MDP)
2. **"Finding the Best Path: Bellman's Insight"** (Session 2 - Bellman Equations)
3. **"From Perfect Planning to Learning Through Trial"** (Session 3 - DP + Q-Learning)
4. **"Learning from Every Step: The TD Revolution"** (Session 4 - TD + PPO intro)

### **GenAI Track Blogs (Sessions 5-8)**

5. **"The Attention Revolution: How AI Learned to Focus"** (Session 5 - Attention)
6. **"Building Blocks of Language Models: Transformer Deep Dive"** (Session 6 - Architecture)
7. **"Teaching Machines to Speak: Language Modeling Fundamentals"** (Session 7 - LM + Tokenization)
8. **"The Mathematics of Intelligence: Understanding Scaling"** (Session 8 - Scaling Laws)

### **Convergence Track Blogs (Sessions 9-12)**

9. **"Teaching AI Human Values: The RLHF Revolution"** (Session 9 - Preference Learning)
10. **"RLHF in Practice: Training Language Models with PPO"** (Session 10 - PPO for LMs)
11. **"Beyond RLHF: DPO, Constitutional AI, and the Future"** (Session 11 - Advanced Methods)
12. **"RLHF Mastery: From Bellman Equations to Constitutional AI"** (Session 12 - Integration)

---

## Sprint Management

### **Weekly Progression**

- **Week 1 (Aug 5-11)**: RL Foundations (MDP, Bellman, DP, Q-Learning)
- **Week 2 (Aug 12-18)**: RL to GenAI Transition (TD Methods + Attention)
- **Week 3 (Aug 19-25)**: GenAI Foundations (Transformers, Language Modeling)
- **Week 4 (Aug 26-Sep 1)**: Advanced GenAI + RLHF Preparation
- **Week 5 (Sep 2-8)**: RLHF Implementation and Mastery
- **Week 6 (Sep 9-12)**: Advanced Methods and Integration

### **Success Metrics**

- **Technical**: Working implementations for each algorithm/architecture
- **Conceptual**: Can explain and teach each concept clearly
- **Integration**: Clear understanding of how RL and GenAI converge in RLHF
- **Practical**: Can implement full RLHF pipeline from scratch
- **Communication**: 12 comprehensive blogs documenting the journey

### **Resource Progression**

**Basic → Intermediate → Advanced pattern maintained throughout:**

- **Foundational papers** (original research)
- **Modern tutorials** (practical implementation)
- **Video explanations** (visual understanding)
- **Hands-on coding** (implementation mastery)
- **Cutting-edge research** (future directions)

This sprint takes you from basic RL concepts all the way to state-of-the-art preference optimization methods, with each domain mastered separately before they naturally converge in RLHF.