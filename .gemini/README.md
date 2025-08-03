# Gemini CLI Custom Commands Setup

This directory contains custom command configurations for the Gemini CLI to enhance the learning management workflow for Reinforcement Learning and Generative AI studies.

## Structure

```
.gemini/
├── README.md                    # This file - setup documentation
└── commands/
    └── learning_commands.toml   # Custom command definitions
```

## Custom Commands

The `learning_commands.toml` file defines custom commands that streamline the learning process. These commands are based on the patterns documented in `CUSTOM_COMMANDS_REFERENCE.md` and adapted for TOML configuration format.

### Core Commands Available

- **Learning Management**
  - `/add-resource` (`/ar`) - Add new learning materials
  - `/plan-sprint` (`/ps`) - Generate learning sprints
  - `/session-prep` (`/sp`) - Prepare study sessions
  - `/session-complete` (`/sc`) - Complete and assess sessions
  - `/progress` (`/p`) - Check learning progress

- **Specialized Functions**
  - `/blog-ready` (`/br`) - Check readiness for blogging
  - `/forge-ready` (`/fr`) - Prepare NotebookLM materials
  - `/sprint-adjust` - Modify current sprint
  - `/milestone-check` - Verify readiness for advanced topics

- **Advanced Operations**
  - `/learning-path` - Show path to specific goals
  - `/review-schedule` - Generate spaced repetition schedule
  - `/sync-progress` - Align all tracking documents

## Usage with Gemini CLI

1. **Install Gemini CLI**: Follow the official installation guide
2. **Authentication**: Login with your Google account or API key
3. **Commands**: Use the custom commands directly in your terminal

```bash
# Example usage
gemini /add-resource paper https://arxiv.org/abs/2305.18290 "DPO: Direct Preference Optimization"
gemini /plan-sprint 2-week
gemini /session-prep 1.2
gemini /progress detailed
```

## Configuration Details

### Command Structure
Each command in the TOML file includes:
- `name` - Command identifier
- `aliases` - Short forms (e.g., `/ar` for `/add-resource`)
- `description` - Purpose and functionality
- `usage` - Syntax and parameters
- `parameters` - Typed parameters with validation
- `action` - Internal function mapping
- `example` - Usage demonstration

### Settings
- **Learning Session Duration**: 120 minutes
- **Break Duration**: 15 minutes  
- **Max Daily Clusters**: 3
- **Confidence Threshold for Blog**: "high"
- **Spaced Repetition**: 1, 3, 7, 14, 30 day intervals

### Workspace Integration
Commands are aware of the file structure:
- `MASTER_LEARNING_ROADMAP.md` - Progress tracking
- `SPRINT_PLANNER.md` - Current sprint details
- `KNOWLEDGE_FORGE_PROMPTS.md` - NotebookLM guides
- `LIVING_RESOURCES.md` - Resource collection
- `CUSTOM_COMMANDS_REFERENCE.md` - Documentation

## Customization

To modify or extend the commands:

1. Edit `commands/learning_commands.toml`
2. Add new command sections following the existing pattern
3. Update parameter validation and options as needed
4. Test with Gemini CLI

## Integration with GEMINI.md

The custom commands work in conjunction with the agent configuration in `GEMINI.md`, which defines:
- Learning philosophy and methodology
- Context awareness and preferences  
- Adaptive learning features
- Success metrics and emergency protocols

This creates a comprehensive learning assistance system optimized for technical education in AI/ML domains.

## Notes

- Commands use cluster ID pattern: `\d+\.\d+` (e.g., "1.2", "2.3")
- Confidence levels: "low", "medium", "high", "mastered"
- Resource types: "course", "paper", "video", "book", "blog"
- Focus areas: "reinforcement-learning", "generative-ai", "transformers", "rlhf"

For detailed command documentation, see `CUSTOM_COMMANDS_REFERENCE.md` in the root directory. 