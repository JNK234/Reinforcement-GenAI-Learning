# Living Resources Document - RL & GenAI Mastery
*Last Updated: 2025-08-03*

> 📌 **This is a living document**: Add new resources as you discover them. Star (⭐) the most valuable ones.

## Table of Contents
- [Core University Courses](#core-university-courses)
- [Reinforcement Learning Resources](#reinforcement-learning-resources)
- [Generative AI & LLM Resources](#generative-ai--llm-resources)
- [Mathematical Foundations](#mathematical-foundations)
- [Implementation Resources](#implementation-resources)
- [Blogs & Tutorials](#blogs--tutorials)
- [Research Papers](#research-papers)
- [YouTube Channels & Playlists](#youtube-channels--playlists)
- [Communities & Forums](#communities--forums)
- [Tools & Frameworks](#tools--frameworks)
- [Books](#books)
- [Datasets & Environments](#datasets--environments)
- [Industry Applications](#industry-applications)

---

## Core University Courses

### Reinforcement Learning Courses

#### ⭐ Stanford CS234: Reinforcement Learning (Winter 2025)
- **Website**: https://web.stanford.edu/class/cs234/index.html
- **Instructor**: Emma Brunskill
- **Videos**: [YouTube Playlist](#) (Add when available)
- **Materials**: Slides, Assignments, Project guidelines
- **Prerequisites**: Python, Calculus, Linear Algebra, Basic ML
- **Key Topics**: MDPs, Q-Learning, Policy Gradients, Exploration

#### ⭐ UC Berkeley CS285: Deep Reinforcement Learning
- **Website**: https://rail.eecs.berkeley.edu/deeprlcourse/
- **Instructor**: Sergey Levine
- **Videos**: [YouTube Playlist](https://www.youtube.com/playlist?list=PL_iWQOsE6TfX7MaC6C3HcdOf1g337dlC9)
- **Materials**: Lecture slides, Homework assignments
- **Key Topics**: Deep Q-Learning, Actor-Critic, Model-Based RL, Offline RL

#### UC Berkeley CS287: Advanced Robotics (Fall 2019)
- **Website**: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/
- **Instructor**: Pieter Abbeel
- **Materials**: Lecture notes, Assignments
- **Key Topics**: Optimal Control, Motion Planning, SLAM, Learning for Robotics

#### Princeton ECE524: Foundations of Reinforcement Learning
- **Website**: https://sites.google.com/view/cjin/teaching/ece524
- **Instructor**: Chi Jin
- **Materials**: Lecture notes, Problem sets
- **Key Topics**: RL Theory, Regret Bounds, Game Theory, PAC Learning

#### ⭐ Stanford CS336: Language Modeling from Scratch (Spring 2025)
- **Website**: https://stanford-cs336.github.io/spring2025/
- **Instructors**: Tatsunori Hashimoto, Percy Liang
- **Materials**: 5 comprehensive assignments
- **Key Topics**: Transformers, Distributed Training, Scaling Laws, Alignment

### Additional University Courses

#### MIT 6.7900: Machine Learning (Former 6.867)
- **Website**: http://www.mit.edu/~9.520/fall23/
- **OpenCourseWare**: https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/
- **Key Topics**: Foundational ML theory

#### CMU 10-703: Deep Reinforcement Learning
- **Website**: https://cmudeeprl.github.io/703website/
- **Instructors**: Katerina Fragkiadaki, Tom Mitchell
- **Key Topics**: Deep RL, Robotics applications

#### DeepMind x UCL: Reinforcement Learning Lecture Series
- **Videos**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)
- **Instructor**: David Silver
- **Key Topics**: Complete RL course from DeepMind researcher

---

## Reinforcement Learning Resources

### Online Courses & MOOCs

#### ⭐ Spinning Up in Deep RL (OpenAI)
- **Website**: https://spinningup.openai.com/
- **Type**: Tutorial + Code
- **Topics**: Key algorithms, Implementations in PyTorch/TensorFlow

#### ⭐ Hugging Face Deep RL Course
- **Website**: https://huggingface.co/learn/deep-rl-course/en/unit0/introduction
- **Type**: Free comprehensive course with hands-on projects
- **Topics**: Q-Learning to PPO, Unity ML-Agents, Multi-Agent RL
- **Time**: 24-32 hours (3-4 hours/week per unit)
- **Difficulty**: Beginner to expert progression
- **Practical Focus**: 8 environments, certification available
- **Libraries**: Stable Baselines3, CleanRL integration
> 💡 **Added 2025-08-03**: Excellent practical complement to academic materials. Could serve as alternative primary track for implementation-focused learners.

#### Deep RL Bootcamp (Berkeley 2017)
- **Videos**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLNvtBXW0ijMRjh5_xKYjXHLuVMkdBPAdC)
- **Materials**: Slides and labs

### Tutorials & Implementations

#### ⭐ CleanRL
- **GitHub**: https://github.com/vwxyzjn/cleanrl
- **Description**: High-quality single-file implementations of RL algorithms
- **Algorithms**: DQN, PPO, SAC, TD3, and more

#### Stable Baselines3
- **GitHub**: https://github.com/DLR-RM/stable-baselines3
- **Docs**: https://stable-baselines3.readthedocs.io/
- **Description**: Reliable implementations of RL algorithms

#### RLlib (Ray)
- **Website**: https://docs.ray.io/en/latest/rllib/index.html
- **Description**: Scalable RL library for production

#### ⭐ The 37 Implementation Details of PPO
- **Blog**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **Description**: Deep dive into PPO implementation nuances

### RL Theory Resources

#### Algorithms for Reinforcement Learning
- **Author**: Csaba Szepesvári
- **PDF**: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
- **Type**: Concise theoretical introduction

#### Reinforcement Learning: Theory and Algorithms
- **Authors**: Alekh Agarwal, Nan Jiang, Sham M. Kakade, Wen Sun
- **Website**: https://rltheorybook.github.io/
- **Type**: Modern RL theory textbook

---

## Generative AI & LLM Resources

### Courses & Tutorials

#### ⭐ Andrej Karpathy's Neural Networks: Zero to Hero
- **YouTube**: [Playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **GitHub**: https://github.com/karpathy/nn-zero-to-hero
- **Projects**: makemore, nanoGPT

#### Fast.ai Practical Deep Learning Part 2
- **Website**: https://www.fast.ai/posts/part2-2022.html
- **Focus**: NLP and Transformers

#### HuggingFace NLP Course
- **Website**: https://huggingface.co/learn/nlp-course/chapter1/1
- **Topics**: Transformers, Fine-tuning, Deployment

#### Full Stack LLM Bootcamp
- **Website**: https://fullstackdeeplearning.com/llm-bootcamp/
- **Videos**: Free lectures on building LLM applications

### Implementation Guides

#### ⭐ The Illustrated Transformer
- **Blog**: https://jalammar.github.io/illustrated-transformer/
- **Author**: Jay Alammar
- **Description**: Visual guide to transformer architecture

#### nanoGPT
- **GitHub**: https://github.com/karpathy/nanoGPT
- **Description**: Minimal GPT implementation for learning

#### x-transformers
- **GitHub**: https://github.com/lucidrains/x-transformers
- **Description**: Modular transformer implementations

#### ⭐ LLM Training/Fine-tuning Guides
- **Weights & Biases LLM Course**: https://www.wandb.courses/courses/training-fine-tuning-LLMs
- **Lambda Labs LLM Training**: https://lambdalabs.com/blog/demystifying-gpt-3

### Alignment & Safety Resources

#### OpenAI Alignment Research
- **Blog**: https://openai.com/alignment/
- **Papers**: InstructGPT, WebGPT, Safety research

#### Anthropic AI Safety
- **Website**: https://www.anthropic.com/research
- **Papers**: Constitutional AI, RLHF research

#### Center for AI Safety
- **Website**: https://www.safe.ai/
- **Resources**: Research, courses, reading lists

---

## Mathematical Foundations

### Optimization

#### ⭐ Convex Optimization (Boyd & Vandenberghe)
- **Book**: https://web.stanford.edu/~boyd/cvxbook/
- **Course**: Stanford EE364a
- **Videos**: [YouTube](https://www.youtube.com/playlist?list=PL3940DD956CDF0622)

#### Optimization for Machine Learning
- **Course**: MIT 6.881
- **Materials**: https://people.csail.mit.edu/stefje/fall15/index.html

### Probability & Statistics

#### Probability Theory: The Logic of Science
- **Author**: E.T. Jaynes
- **PDF**: https://bayes.wustl.edu/etj/prob/book.pdf

#### High-Dimensional Probability
- **Author**: Roman Vershynin
- **Website**: https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html

### Linear Algebra

#### ⭐ MIT 18.06: Linear Algebra
- **Instructor**: Gilbert Strang
- **OCW**: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- **Videos**: Complete lecture series

#### The Matrix Cookbook
- **PDF**: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
- **Description**: Quick reference for matrix operations

---

## Implementation Resources

### Project Ideas & Datasets

#### RL Environments
- **OpenAI Gym**: https://github.com/openai/gym
- **PettingZoo** (Multi-agent): https://pettingzoo.farama.org/
- **MuJoCo**: https://mujoco.org/
- **Unity ML-Agents**: https://unity.com/products/machine-learning-agents

#### LLM Datasets
- **The Pile**: https://pile.eleuther.ai/
- **C4**: https://www.tensorflow.org/datasets/catalog/c4
- **RedPajama**: https://github.com/togethercomputer/RedPajama-Data

### Compute Resources

#### Free GPU Options
- **Google Colab**: https://colab.research.google.com/
- **Kaggle Kernels**: https://www.kaggle.com/
- **Paperspace Gradient**: https://gradient.run/free-gpu

#### Cloud Platforms
- **Lambda Labs**: https://lambdalabs.com/
- **Vast.ai**: https://vast.ai/
- **RunPod**: https://www.runpod.io/

---

## Blogs & Tutorials

### RL Blogs

#### ⭐ Lil'Log (Lilian Weng)
- **URL**: https://lilianweng.github.io/
- **Topics**: Comprehensive RL and ML posts

#### The Gradient
- **URL**: https://thegradient.pub/
- **Topics**: ML/AI research perspectives

#### Distill.pub
- **URL**: https://distill.pub/
- **Topics**: Interactive ML visualizations

### LLM/GenAI Blogs

#### ⭐ Sebastian Ruder's Blog
- **URL**: https://ruder.io/
- **Topics**: NLP, Transfer Learning

#### The Transformer Family (Lilian Weng)
- **URL**: https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
- **Topics**: Comprehensive transformer variants

#### Chip Huyen's Blog
- **URL**: https://huyenchip.com/
- **Topics**: MLOps, LLMs in production

---

## Research Papers

### Foundational RL Papers

#### ⭐ Must-Read Classic Papers
1. **Q-Learning** (Watkins, 1989): [Link](https://link.springer.com/article/10.1007/BF00992698)
2. **Policy Gradient Theorem** (Sutton et al., 1999): [Link](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
3. **DQN** (Mnih et al., 2015): [Link](https://www.nature.com/articles/nature14236)
4. **A3C** (Mnih et al., 2016): [Link](https://arxiv.org/abs/1602.01783)
5. **PPO** (Schulman et al., 2017): [Link](https://arxiv.org/abs/1707.06347)
6. **SAC** (Haarnoja et al., 2018): [Link](https://arxiv.org/abs/1801.01290)

### Foundational LLM Papers

#### ⭐ Transformer & Language Model Papers
1. **Attention Is All You Need** (2017): [Link](https://arxiv.org/abs/1706.03762)
2. **BERT** (2018): [Link](https://arxiv.org/abs/1810.04805)
3. **GPT-3** (2020): [Link](https://arxiv.org/abs/2005.14165)
4. **Scaling Laws** (Kaplan et al., 2020): [Link](https://arxiv.org/abs/2001.08361)
5. **InstructGPT** (2022): [Link](https://arxiv.org/abs/2203.02155)
6. **Constitutional AI** (2022): [Link](https://arxiv.org/abs/2212.08073)

### Paper Collections

#### Papers With Code
- **RL Section**: https://paperswithcode.com/area/reinforcement-learning
- **NLP Section**: https://paperswithcode.com/area/natural-language-processing

#### Awesome RL Papers
- **GitHub**: https://github.com/aikorea/awesome-rl

---

## YouTube Channels & Playlists

### RL Focused

#### ⭐ Two Minute Papers
- **Channel**: https://www.youtube.com/c/K%C3%A1rolyZsolnai
- **Focus**: AI/ML paper summaries

#### Yannic Kilcher
- **Channel**: https://www.youtube.com/c/YannicKilcher
- **Focus**: Deep technical paper reviews

#### AI Coffee Break with Letitia
- **Channel**: https://www.youtube.com/c/AICoffeeBreak
- **Focus**: ML concepts explained simply

### LLM/GenAI Focused

#### ⭐ Andrej Karpathy
- **Channel**: https://www.youtube.com/@AndrejKarpathy
- **Focus**: Neural networks from scratch

#### Jeremy Howard (fast.ai)
- **Channel**: https://www.youtube.com/user/howardjeremyp
- **Focus**: Practical deep learning

---

## Communities & Forums

### Discord Servers
- **Eleuther AI**: https://discord.gg/eleutherai
- **Hugging Face**: https://discord.gg/huggingface
- **Fast.ai**: https://discord.gg/56kgNXD

### Reddit Communities
- **/r/MachineLearning**: https://www.reddit.com/r/MachineLearning/
- **/r/reinforcementlearning**: https://www.reddit.com/r/reinforcementlearning/
- **/r/LocalLLaMA**: https://www.reddit.com/r/LocalLLaMA/

### Slack/Forums
- **MLOps Community**: https://mlops.community/
- **Papers With Code**: https://paperswithcode.com/

---

## Tools & Frameworks

### RL Frameworks
- **Gymnasium** (OpenAI Gym successor): https://gymnasium.farama.org/
- **TorchRL**: https://pytorch.org/rl/
- **TF-Agents**: https://www.tensorflow.org/agents

### LLM Frameworks
- **LangChain**: https://python.langchain.com/
- **LlamaIndex**: https://www.llamaindex.ai/
- **vLLM**: https://github.com/vllm-project/vllm

### Experiment Tracking
- **Weights & Biases**: https://wandb.ai/
- **MLflow**: https://mlflow.org/
- **TensorBoard**: https://www.tensorflow.org/tensorboard

---

## Books

### RL Books

#### ⭐ Reinforcement Learning: An Introduction
- **Authors**: Sutton & Barto
- **Edition**: 2nd (2018)
- **Free PDF**: http://incompleteideas.net/book/the-book.html

#### Deep Reinforcement Learning Hands-On
- **Author**: Maxim Lapan
- **Publisher**: Packt
- **Focus**: Practical implementations

### ML/DL Books

#### ⭐ Deep Learning
- **Authors**: Goodfellow, Bengio, Courville
- **Free**: https://www.deeplearningbook.org/

#### Pattern Recognition and Machine Learning
- **Author**: Christopher Bishop
- **Focus**: Statistical ML foundations

### Math Books

#### ⭐ Mathematics for Machine Learning
- **Authors**: Deisenroth, Faisal, Ong
- **Free PDF**: https://mml-book.github.io/

---

## Datasets & Environments

### RL Benchmarks
- **Atari**: Classic control tasks
- **MuJoCo**: Continuous control
- **Procgen**: Procedurally generated games
- **Meta-World**: Multi-task robotics

### LLM Benchmarks
- **GLUE**: General Language Understanding
- **SuperGLUE**: Harder language tasks
- **MMLU**: Massive Multitask Language Understanding
- **HumanEval**: Code generation

---

## Industry Applications

### RL in Production
- **Recommendation Systems**: YouTube, Netflix
- **Robotics**: Boston Dynamics, Tesla
- **Game AI**: AlphaGo, OpenAI Five
- **Finance**: Trading, Portfolio optimization

### LLMs in Production
- **Code Generation**: GitHub Copilot, Cursor
- **Search**: Perplexity, You.com
- **Writing**: Jasper, Copy.ai
- **Customer Service**: Intercom, Zendesk

---

## How to Use This Document

1. **Star (⭐) resources** as you find them particularly valuable
2. **Add new sections** as you discover new resource categories
3. **Update links** when you find better/newer versions
4. **Add personal notes** in blockquotes under resources:
   > 💡 Personal note: This resource was particularly helpful for understanding...

5. **Track progress** with checkboxes:
   - [ ] Not started
   - [🔄] In progress  
   - [✅] Completed

6. **Date your additions** to track when resources were added
7. **Create cross-references** between related resources

---

## Contributing to This Document

When adding new resources:
1. Include title, URL, author/source
2. Add brief description of why it's valuable
3. Categorize appropriately
4. Note prerequisites if any
5. Add personal experience/notes if applicable

Remember: Quality > Quantity. Focus on resources that truly helped you understand concepts deeply.