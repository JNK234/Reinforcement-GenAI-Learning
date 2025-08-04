# Custom Commands Reference
*Streamlined Learning Management*

## Core Learning Commands

### `/add-resource [type] [url/details]`
**Purpose**: Add new learning material to roadmap  
**Types**: `course`, `paper`, `video`, `book`, `blog`  
**Example**: `/add-resource paper https://arxiv.org/abs/2305.18290 "DPO: Direct Preference Optimization"`  
**Action**: Analyzes content, suggests cluster placement, updates roadmap

### `/plan-sprint [duration]`
**Purpose**: Generate next learning sprint  
**Options**: `1-week`, `2-week` (default)  
**Example**: `/plan-sprint 2-week`  
**Action**: Creates session schedule, prepares materials, sets blog targets

### `/session-prep [cluster-id]`
**Purpose**: Prepare for learning session  
**Example**: `/session-prep 1.2`  
**Action**: Lists materials, provides Knowledge Forge prompts, sets expectations

### `/session-complete [cluster-id] [confidence]`
**Purpose**: Wrap up learning session  
**Confidence**: `low`, `medium`, `high`, `mastered`  
**Example**: `/session-complete 1.2 high`  
**Action**: Updates progress, identifies gaps, prepares blog pipeline

### `/progress [view]`
**Purpose**: Check learning progress  
**Views**: `summary` (default), `detailed`, `timeline`  
**Example**: `/progress detailed`  
**Action**: Shows clusters completed, blogs published, what's next

## Specialized Commands

### `/blog-ready [cluster-id]`
**Purpose**: Check if ready for blogging  
**Example**: `/blog-ready 1.3`  
**Action**: Verifies topic coverage, suggests blog scope and edition

### `/forge-ready [cluster-id]`
**Purpose**: Get NotebookLM session materials  
**Example**: `/forge-ready 2.1`  
**Action**: Lists upload materials, provides prompts, sets success criteria

### `/sprint-adjust [reason]`
**Purpose**: Modify current sprint  
**Reasons**: `behind`, `ahead`, `new-priority`, `time-constraint`  
**Example**: `/sprint-adjust behind`  
**Action**: Redistributes topics, adjusts timeline, updates expectations

### `/milestone-check [target]`
**Purpose**: Verify readiness for advanced topics  
**Targets**: `deep-rl`, `transformers`, `rlhf`, `research`  
**Example**: `/milestone-check deep-rl`  
**Action**: Checks prerequisites, assesses confidence, recommends review

## Quick Commands (Shortcuts)

- `/ar` → `/add-resource`
- `/ps` → `/plan-sprint`  
- `/sp` → `/session-prep`
- `/sc` → `/session-complete`
- `/p` → `/progress`
- `/br` → `/blog-ready`
- `/fr` → `/forge-ready`

## Advanced Commands

### `/learning-path [goal]`
**Purpose**: Show path to specific goal  
**Example**: `/learning-path "implement PPO from scratch"`  
**Action**: Maps required clusters, estimates timeline, highlights critical topics

### `/review-schedule`
**Purpose**: Generate spaced repetition schedule  
**Action**: Identifies topics needing review based on completion dates and confidence

### `/sync-progress`
**Purpose**: Align all tracking documents  
**Action**: Updates master roadmap, refreshes sprint planner, resolves conflicts

## Command Usage Patterns

### **Starting New Sprint**:
```
/plan-sprint 2-week
/session-prep 1.1
```

### **During Learning Session**:
```
/forge-ready 1.1
[Complete NotebookLM session]
/session-complete 1.1 high
```

### **Weekly Review**:
```
/progress detailed
/review-schedule
/sprint-adjust [if needed]
```

### **Adding New Resources**:
```
/add-resource course https://new-course.com
[Review suggestions]
/plan-sprint [if major changes needed]
```

### **Blog Pipeline**:
```
/blog-ready 1.2
[Write blog post]
/progress summary [to see updated counts]
```

## Command Output Examples

### `/progress summary` Output:
```
Learning Progress Dashboard
Clusters Completed: 3/24 | Blogs Published: 2/24
Current Focus: Model-Free Methods
Next Up: Cluster 2.2 - Temporal Difference Learning
This Week: On track for 2 sessions
```

### `/session-prep 1.3` Output:
```
Session 3: Cluster 1.3 - Bellman Equations

Materials to Upload:
- CS234 Lecture 2 Bellman section (PDF)
- Sutton & Barto Ch 3.5-3.8 (pages 58-75)
- David Silver Bellman slides (PDF)

Knowledge Forge Prompts: Ready ✅
Time Allocation: 30min prep + 90min session + 30min notes + 30min blog
Success Criteria: Can derive Bellman equations, understand recursive structure
```

### `/blog-ready 1.2` Output:
```
Cluster 1.2 Blog Assessment:

✅ Topic Coverage: Complete
✅ Concept Clarity: High confidence
✅ Cross-references: Available
✅ Edition Readiness: Both Scientist & Manager

Suggested Blog: "The Mathematical Foundation of AI Decision Making"
Topics: MDP formalism, states/actions/rewards, policy concepts
Reading Time: ~12-15 minutes
Recommended Edition: Scientist (mathematical focus)
```

## Tips for Effective Command Usage

1. **Be Specific**: Use exact cluster IDs (e.g., `1.2` not `bellman`)
2. **Regular Check-ins**: Use `/progress` weekly to stay on track
3. **Prep Early**: Run `/session-prep` the day before learning sessions
4. **Honest Assessment**: Use accurate confidence levels in `/session-complete`
5. **Adjust Proactively**: Use `/sprint-adjust` as soon as you notice timing issues

## Custom Command Creation

Want to create your own commands? Use this pattern:

```
/create-command [name] [description]
Example: /create-command weekly-review "Show progress + plan next week"
```

Remember: These commands are designed to reduce cognitive overhead and keep you focused on learning, not managing your learning system.