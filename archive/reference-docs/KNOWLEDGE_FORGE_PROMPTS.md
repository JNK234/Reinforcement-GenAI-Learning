# Generic Knowledge Forge Prompts - 3 Stage System

## Stage 1: Initial Scan & Overview Prompt

```
You are my expert technical tutor. I've uploaded materials about [TOPIC].

Your first task:
1. Scan all materials and identify 5-7 core concepts
2. List any key equations or algorithms present
3. Don't explain yet - just give me the high-level map

End with: "Ready for deep dive! What would you like to explore first?"
```

## Stage 2: Interactive Q&A Prompts

### My Question Starters:
```
"Explain [CONCEPT] - start with intuition, then formalize"
"Walk me through [EQUATION] step by step"
"I'm confused about [X] - use an analogy"
"How does [A] relate to [B]?"
"Show me [ALGORITHM] in pseudocode"
"What problem does [CONCEPT] solve?"
"When does [METHOD] fail?"
```

### Follow-up When Confused:
```
"Still unclear - try a different angle"
"Give me a minimal example"
"What's the simplest case?"
"Compare with something I might know"
"Break this down further"
```

## Stage 3: Synthesis Command

```
I'm ready for comprehensive notes. Please synthesize everything:

Structure:
1. Logical flow from basics to advanced
2. Integrate ALL my confusion points with clarifications
3. Include all math (LaTeX), code, and examples we discussed
4. Show connections between concepts
5. End with 5-7 test questions

Make it read as one flowing document, not Q&A format.
```

---

## Progressive Learning Prompts by Topic

### Foundation Level Prompts

#### 1. First Concepts (MDPs)
```
I'm learning about MDPs. Help me understand:
1. What problem do MDPs solve?
2. Core components (states, actions, rewards, transitions)
3. Simple gridworld example
4. Why this matters for sequential decisions

Build from intuition → formal definition → practical example.
```

#### 2. Building on Foundations (Bellman Equations)
```
Now that I understand MDPs, explain Bellman equations:
1. Connect to MDP components I already know
2. Intuition before math
3. Derive value function step-by-step
4. Show how it enables planning

Assume I understand: [previous concepts]
New concepts to introduce: [current topic]
```

#### 3. Algorithm Introduction (Value Iteration)
```
Building on MDPs and Bellman equations, teach me value iteration:
1. Why we need iterative methods
2. Algorithm pseudocode with comments
3. Convergence intuition
4. When it works/fails

Prerequisites I have: MDPs, Bellman equations
Goal: Implement this algorithm
```

### Intermediate Level Prompts

#### 4. Model-Free Methods (Q-Learning)
```
Transition me from planning (with models) to learning (without models):
1. Why model-free matters
2. Q-learning as online value iteration
3. Exploration vs exploitation
4. Implementation considerations

Bridge from: Dynamic programming
Bridge to: Deep Q-learning
```

#### 5. Function Approximation
```
Scale up from tabular to continuous:
1. Why tables fail at scale
2. Neural networks as value functions
3. What breaks and why (theory)
4. Practical fixes (experience replay, target networks)

I understand: Tabular Q-learning
Help me understand: Deep Q-Networks
```

### Advanced Level Prompts

#### 6. Policy Gradients
```
Shift my thinking from value-based to policy-based:
1. New optimization perspective
2. Gradient derivation (build intuition first)
3. Why variance is the enemy
4. Connection to supervised learning

Compare with: Q-learning approach
Emphasize: Practical advantages/disadvantages
```

#### 7. Actor-Critic
```
Combine what I know about values and policies:
1. Best of both worlds intuition
2. How critic helps actor
3. PPO as practical implementation
4. Stability improvements

Synthesize: Value methods + Policy methods
Focus on: Why this dominates in practice
```

---

## Concept Building Sequences

### Sequence 1: RL Foundations
```
Learn these in order:
1. Sequential decisions → MDPs
2. MDPs → Bellman equations  
3. Bellman → Dynamic programming
4. Known models → Unknown models
5. Tabular → Function approximation
6. Values → Policies
7. Separate → Actor-Critic

Each concept directly enables the next.
```

### Sequence 2: Deep Learning Integration
```
Learn these in order:
1. Supervised learning basics
2. Neural network as function approximator
3. Q-function approximation challenges
4. Stability tricks (replay, targets)
5. Policy networks
6. End-to-end training

Bridge ML knowledge to RL applications.
```

### Sequence 3: Transformers & LLMs
```
Learn these in order:
1. Sequence modeling problem
2. Attention as information routing
3. Self-attention mechanics
4. Multi-head for representation
5. Transformer architecture
6. Scaling properties
7. Emergent capabilities

Build from need → solution → implementation → properties.
```

---

## Quick Reference Prompts

### For Confusion Points
```
I'm confused about [SPECIFIC CONCEPT]. Please:
1. Explain the intuition in simple terms
2. Give a minimal example
3. Contrast with what I might be mixing it up with
4. Show where this fits in the bigger picture
```

### For Mathematical Understanding
```
Walk me through [EQUATION/THEOREM]:
1. What problem does this solve?
2. Explain each term's meaning
3. Derive it step-by-step
4. Show a numerical example
5. When does this break down?
```

### For Implementation
```
Help me implement [ALGORITHM]:
1. Pseudocode with clear comments
2. Key implementation details that matter
3. Common bugs to avoid
4. How to verify it's working
5. Connection to theory
```

### For Synthesis
```
I'm ready to synthesize [TOPIC]. Create notes that:
1. Flow from basics I learned to advanced concepts
2. Integrate all my confusion points with clarifications
3. Include worked examples
4. Connect to prerequisites and future topics
5. End with 5 test questions
```

---

## Progressive Difficulty Checklist

### Beginner Topics (Start Here)
- [ ] What is RL? (vs supervised learning)
- [ ] Multi-armed bandits (simplest RL)
- [ ] Markov Decision Processes
- [ ] Dynamic Programming (when you know everything)

### Intermediate Topics
- [ ] Monte Carlo methods
- [ ] Temporal Difference learning
- [ ] Q-Learning
- [ ] Function approximation basics
- [ ] Deep Q-Networks

### Advanced Topics  
- [ ] Policy gradient theorem
- [ ] Actor-Critic methods
- [ ] PPO/TRPO
- [ ] Model-based RL
- [ ] Offline RL
- [ ] Multi-agent RL

### Expert Topics
- [ ] Meta-RL
- [ ] Hierarchical RL
- [ ] RL theory (regret bounds, PAC)
- [ ] Inverse RL
- [ ] RL + LLMs

---

## Template for Progressive Sessions

### Session Structure
```
Session Goal: Master [CONCEPT] building on [PREREQUISITES]

1. Upload to NotebookLM:
   - One foundational PDF
   - One practical/implementation resource
   - Previous session's synthesis notes

2. Opening prompt:
   "I understand [PREREQUISITES]. Now teach me [NEW CONCEPT] by:
   - Connecting to what I know
   - Building intuition before formalism  
   - Showing practical importance
   - Preparing me for [NEXT CONCEPT]"

3. Follow-up prompts as needed

4. Synthesis prompt when ready

5. Export and create atomic notes
```

### Tracking Progress
After each session, note:
- Concepts mastered: ✓
- Confusion points resolved: ✓
- Ready for next topic: ✓
- Blog post potential: ✓

---

## Manager vs Scientist Prompts

### Manager Perspective
```
Explain [TECHNICAL CONCEPT] for strategic decisions:
1. Business problem it solves
2. ROI and resource requirements
3. Team/skills needed
4. Risk and limitations
5. Competitive advantage
No math needed, focus on outcomes.
```

### Scientist Perspective  
```
Deep dive into [TECHNICAL CONCEPT]:
1. Mathematical foundations
2. Algorithmic details
3. Theoretical guarantees
4. Implementation nuances
5. Research frontiers
Include all math, proofs, and code.
```

Remember: Keep prompts focused and build knowledge progressively. Each session should clearly connect to previous learning and prepare for what's next.