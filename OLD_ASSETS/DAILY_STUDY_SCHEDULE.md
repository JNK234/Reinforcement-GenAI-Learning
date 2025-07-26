# Daily Study Schedule - RL & GenAI Mastery

## Week 1-2: MDP Fundamentals

### Day 1 (Monday): Introduction to Sequential Decisions
**Morning (2 hours)**
- [ ] Watch: [CS234 Lecture 1](https://www.youtube.com/watch?v=FgzM3zpZ55o) - Introduction to RL
- [ ] Read: Sutton & Barto Ch 1 (Pages 1-24) [PDF](http://incompleteideas.net/book/RLbook2020.pdf#page=17)
- [ ] NotebookLM Session: Upload both resources, use foundation prompt

**Afternoon (2 hours)**
- [ ] Read: [What is RL? - OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [ ] Implementation: Simple multi-armed bandit
- [ ] Create atomic notes: "What is RL", "RL vs Supervised Learning"

**Evening (1 hour)**
- [ ] Review notes and create blog outline
- [ ] Update progress in Obsidian

### Day 2 (Tuesday): MDP Components
**Morning (2 hours)**
- [ ] Read: CS234 Lecture 2 Slides [PDF](https://web.stanford.edu/class/cs234/slides/cs234_lecture2.pdf)
- [ ] Watch: [David Silver Lecture 2 - MDPs](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
- [ ] NotebookLM: Upload both, ask about states, actions, rewards

**Afternoon (2 hours)**
- [ ] Read: Sutton & Barto Ch 3.1-3.4 [PDF](http://incompleteideas.net/book/RLbook2020.pdf#page=65)
- [ ] Code: Implement GridWorld MDP
- [ ] Manager perspective: Write "MDPs in Business Decisions"

**Evening (1 hour)**
- [ ] Practice problems from CS234 Assignment 1
- [ ] Create MDP visualization diagrams

### Day 3 (Wednesday): Bellman Equations
**Morning (2 hours)**
- [ ] Read: CS234 Lecture 2 (continued) - Bellman Equations
- [ ] Watch: [MIT 6.034 - Value Functions](https://www.youtube.com/watch?v=IXiHwqVEXGo)
- [ ] NotebookLM: Deep dive on Bellman derivation

**Afternoon (2 hours)**
- [ ] Read: David Silver Slides [PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf)
- [ ] Implementation: Bellman equation solver
- [ ] Create notes: "Bellman Optimality", "Recursive Value"

### Day 4 (Thursday): Dynamic Programming Theory
**Morning (2 hours)**
- [ ] Read: CS234 Lecture 3 [PDF](https://web.stanford.edu/class/cs234/slides/cs234_lecture3.pdf)
- [ ] Read: Sutton & Barto Ch 4.1-4.3 [PDF](http://incompleteideas.net/book/RLbook2020.pdf#page=89)
- [ ] NotebookLM: Upload both, focus on when DP works

**Afternoon (2 hours)**
- [ ] Watch: [CS285 Planning Lecture](https://www.youtube.com/watch?v=d0nVzblvpDI)
- [ ] Read: Bertsekas DP Ch 1 [PDF](http://www.athenasc.com/dpchapter.pdf)
- [ ] Manager blog: "Strategic Planning with Perfect Information"

### Day 5 (Friday): Value & Policy Iteration
**Morning (2 hours)**
- [ ] Code along: [DP Tutorial](https://github.com/dennybritz/reinforcement-learning/tree/master/DP)
- [ ] Implement: Value iteration on larger gridworld
- [ ] Implement: Policy iteration comparison

**Afternoon (2 hours)**
- [ ] Read: CS234 Assignment 2 Solutions
- [ ] Create comparative analysis: VI vs PI
- [ ] Scientists blog draft: "Optimal Planning Algorithms"

**Weekend Review**
- [ ] Consolidate all MDP/DP notes in Obsidian
- [ ] Complete first blog post
- [ ] Self-test with NotebookLM quiz generation

---

## Week 3-4: Model-Free Methods

### Day 6 (Monday): Monte Carlo Methods
**Morning (2 hours)**
- [ ] Read: Sutton & Barto Ch 5 [PDF](http://incompleteideas.net/book/RLbook2020.pdf#page=111)
- [ ] Watch: [CS234 Lecture 4](https://www.youtube.com/watch?v=i0o-ui1N35U)
- [ ] NotebookLM: "From planning to learning"

**Afternoon (2 hours)**
- [ ] Read: CS234 Slides [PDF](https://web.stanford.edu/class/cs234/slides/cs234_lecture4.pdf)
- [ ] Implement: Monte Carlo for Blackjack
- [ ] Notes: "First-visit vs Every-visit MC"

### Day 7 (Tuesday): Temporal Difference Learning
**Morning (2 hours)**
- [ ] Read: Sutton & Barto Ch 6.1-6.3 [PDF](http://incompleteideas.net/book/RLbook2020.pdf#page=129)
- [ ] Watch: [David Silver TD Learning](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- [ ] NotebookLM: TD vs MC comparison

**Afternoon (2 hours)**
- [ ] Read: CS285 TD Slides [PDF](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)
- [ ] Implement: TD(0) algorithm
- [ ] Manager perspective: "Learning from Experience in Business"

### Day 8 (Wednesday): Q-Learning
**Morning (2 hours)**
- [ ] Read: Original Q-Learning Paper [PDF](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
- [ ] Watch: [CS234 Q-Learning Lecture](https://www.youtube.com/watch?v=a0i_bT4ujBE)
- [ ] NotebookLM: Q-learning convergence

**Afternoon (2 hours)**
- [ ] Read: CS234 Lecture 5 [PDF](https://web.stanford.edu/class/cs234/slides/cs234_lecture5.pdf)
- [ ] Implement: Q-learning for GridWorld
- [ ] Implement: Epsilon-greedy exploration

### Day 9 (Thursday): SARSA & Variants
**Morning (2 hours)**
- [ ] Read: Sutton & Barto Ch 6.4-6.6
- [ ] Compare: Q-learning vs SARSA implementations
- [ ] NotebookLM: On-policy vs off-policy

**Afternoon (2 hours)**
- [ ] Implement: Expected SARSA
- [ ] Read: [Comparison of TD methods](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdmethods.html)
- [ ] Create decision tree: "Which TD method when?"

### Day 10 (Friday): Integration & Projects
**Morning (2 hours)**
- [ ] Build: Unified TD agent (Q-learning, SARSA, Expected SARSA)
- [ ] Test on multiple environments
- [ ] Create performance comparison plots

**Afternoon (2 hours)**
- [ ] Write blog: "Model-Free RL: A Practical Guide"
- [ ] Update Obsidian with all TD method notes
- [ ] Plan next phase: Function Approximation

---

## Week 5-6: Deep RL Foundations

### Day 11 (Monday): Function Approximation Intro
**Morning (2 hours)**
- [ ] Read: Sutton & Barto Ch 9 [PDF](http://incompleteideas.net/book/RLbook2020.pdf#page=217)
- [ ] Watch: [CS285 Function Approximation](https://www.youtube.com/watch?v=Vky6bGWkJW0)
- [ ] NotebookLM: "Why neural networks for RL?"

**Afternoon (2 hours)**
- [ ] Read: CS234 Lecture 6 [PDF](https://web.stanford.edu/class/cs234/slides/cs234_lecture6.pdf)
- [ ] Tutorial: [Deep Learning Basics Review](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [ ] Setup: PyTorch environment for Deep RL

### Day 12 (Tuesday): DQN Theory
**Morning (2 hours)**
- [ ] Read: Original DQN Paper [PDF](https://arxiv.org/pdf/1312.5602.pdf)
- [ ] Read: Nature DQN Paper [PDF](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [ ] NotebookLM: Experience replay and target networks

**Afternoon (2 hours)**
- [ ] Watch: [CS285 DQN Lecture](https://www.youtube.com/watch?v=KHx1nqgODCY)
- [ ] Read: DQN Slides [PDF](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf)
- [ ] Manager blog: "From Atari to Real-World AI"

### Day 13 (Wednesday): DQN Implementation
**Morning (2 hours)**
- [ ] Code: [CleanRL DQN](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
- [ ] Study: Implementation details that matter
- [ ] Debug: Common DQN pitfalls

**Afternoon (2 hours)**
- [ ] Tutorial: [HuggingFace DQN](https://huggingface.co/blog/deep-rl-dqn)
- [ ] Implement: DQN for CartPole
- [ ] Visualize: Learning curves, Q-values

### Day 14 (Thursday): DQN Improvements
**Morning (2 hours)**
- [ ] Read: Rainbow DQN [PDF](https://arxiv.org/pdf/1710.02298.pdf)
- [ ] Read: Prioritized Experience Replay [PDF](https://arxiv.org/pdf/1511.05952.pdf)
- [ ] NotebookLM: Which improvements matter most?

**Afternoon (2 hours)**
- [ ] Implement: Double DQN
- [ ] Implement: Dueling DQN
- [ ] Compare: Performance across variants

### Day 15 (Friday): Project - Stock Trading Bot
**Morning (3 hours)**
- [ ] Design: DQN for stock trading
- [ ] Data: Setup market environment
- [ ] Implement: Basic trading agent

**Afternoon (2 hours)**
- [ ] Backtest: Evaluate performance
- [ ] Blog: "DQN in Finance: Lessons Learned"
- [ ] Plan: Policy gradient methods next

---

## Week 7-8: Policy Gradient Methods

### Day 16 (Monday): Policy Gradient Theory
**Morning (2 hours)**
- [ ] Read: Sutton PG Paper [PDF](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [ ] Watch: [CS285 Policy Gradients](https://www.youtube.com/watch?v=S_gwYj1Q-44)
- [ ] NotebookLM: Derive policy gradient theorem

**Afternoon (2 hours)**
- [ ] Read: CS234 Lecture 7 [PDF](https://web.stanford.edu/class/cs234/slides/cs234_lecture7.pdf)
- [ ] Read: [Lil'Log Policy Gradient](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [ ] Notes: "From values to policies"

### Day 17 (Tuesday): REINFORCE
**Morning (2 hours)**
- [ ] Read: CS285 PG Slides [PDF](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-4.pdf)
- [ ] Implement: Vanilla REINFORCE
- [ ] Debug: High variance issues

**Afternoon (2 hours)**
- [ ] Implement: REINFORCE with baseline
- [ ] Compare: Variance reduction impact
- [ ] Manager blog: "Direct Optimization in AI Systems"

### Day 18 (Wednesday): Natural Policy Gradient
**Morning (2 hours)**
- [ ] Read: NPG Paper [PDF](https://papers.nips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- [ ] Watch: [Natural Gradients Explained](https://www.youtube.com/watch?v=bVQJDaZrdVk)
- [ ] NotebookLM: Fisher information matrix

**Afternoon (2 hours)**
- [ ] Read: TRPO Paper [PDF](https://arxiv.org/pdf/1502.05477.pdf)
- [ ] Study: Trust region methods
- [ ] Notes: "Why natural gradients matter"

### Day 19 (Thursday): Actor-Critic Fundamentals
**Morning (2 hours)**
- [ ] Read: CS285 AC Slides [PDF](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf)
- [ ] Watch: [Actor-Critic Lecture](https://www.youtube.com/watch?v=EKqxumCuAAY)
- [ ] NotebookLM: Combine value and policy learning

**Afternoon (2 hours)**
- [ ] Implement: Basic Actor-Critic
- [ ] Implement: A2C (synchronous)
- [ ] Debug: Common AC issues

### Day 20 (Friday): PPO - The Workhorse
**Morning (2 hours)**
- [ ] Read: PPO Paper [PDF](https://arxiv.org/pdf/1707.06347.pdf)
- [ ] Read: [37 PPO Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [ ] NotebookLM: PPO vs TRPO tradeoffs

**Afternoon (3 hours)**
- [ ] Implement: PPO from scratch
- [ ] Study: [CleanRL PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
- [ ] Project: Multi-agent warehouse with PPO

---

## Daily Routine Template

### Morning Block (2-3 hours)
1. **Theory & Concepts**
   - Primary resource (paper/lecture)
   - Secondary resource (slides/notes)
   - NotebookLM session with both

2. **Knowledge Forging**
   - Ask clarifying questions
   - Work through confusions
   - Generate synthesis

### Afternoon Block (2-3 hours)
1. **Implementation**
   - Code the algorithm
   - Debug and verify
   - Run experiments

2. **Perspective Writing**
   - Manager view: Business implications
   - Scientist view: Technical deep dive
   - Create atomic notes

### Evening Block (1 hour)
1. **Integration**
   - Update Obsidian vault
   - Link new concepts
   - Plan next day

2. **Review**
   - Self-test key concepts
   - Update progress tracker
   - Draft blog ideas

---

## Progress Tracking

### Week Targets
- [ ] 5 NotebookLM sessions
- [ ] 3 algorithm implementations  
- [ ] 10+ atomic notes created
- [ ] 1 blog post (manager OR scientist)
- [ ] All exercises from course

### Monthly Goals
- [ ] Complete one full topic cluster
- [ ] Publish 4 blog posts (2 each perspective)
- [ ] Build 1 significant project
- [ ] Contribute to 1 open-source RL/AI project

Remember: Quality > Speed. Master each concept before moving forward.