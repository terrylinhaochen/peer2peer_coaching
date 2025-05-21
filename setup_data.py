import os
import shutil
from pathlib import Path
import json

def setup_data_directories():
    """Set up the necessary data directories and files for the peer2peer system"""
    # Ensure we're in the right directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Create data directory
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create templates directory
    templates_dir = script_dir / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Check for codebook in root directory first
    root_codebook = Path("../data/codebook.txt")
    peer2peer_codebook = data_dir / "codebook.txt"
    
    if root_codebook.exists():
        print(f"Copying {root_codebook} to {peer2peer_codebook}")
        shutil.copy(root_codebook, peer2peer_codebook)
    elif not peer2peer_codebook.exists():
        # Create a minimal codebook if it doesn't exist
        print(f"Creating minimal codebook at {peer2peer_codebook}")
        with open(peer2peer_codebook, "w", encoding="utf-8") as f:
            f.write("""You are an expert in analyzing student regulation gaps. You need to categorize each CAP (Context, Assessment, Plan) note into these three tier 1 categories and their corresponding tier 2 categories:

1. Cognitive: The student lacks skills for approaching problems with an unknown answer
   - Representing problem and solution spaces
   - Assessing risks
   - Critical thinking and argumentation

2. Metacognitive: The student struggles in areas of planning, help-seeking and collaboration
   - Forming feasible plans
   - Planning effective iterations
   - Leveraging resources and seeking help

3. Emotional: The student has regulation issues that affect motivation and learning
   - Fears and anxieties
   - Embracing challenges and learning

Focus primarily on the "Assessment" part of the note when categorizing, as this is the coach's perceived regulation gap.
""")
    
    # Check for tiered_weighted_cases in root directory
    root_cases = Path("../data/tiered_weighted_cases.json")
    peer2peer_cases = data_dir / "tiered_weighted_cases.json"
    
    if root_cases.exists():
        print(f"Copying {root_cases} to {peer2peer_cases}")
        shutil.copy(root_cases, peer2peer_cases)
    elif not peer2peer_cases.exists():
        # Create a minimal case file if it doesn't exist
        print(f"Creating sample case data at {peer2peer_cases}")
        sample_cases = [
            {
                "id": "01",
                "gap_text": "Not thinking carefully about what is needed given risks at this moment",
                "other_content": "Title: Planning is too rushed\nContext: Sprint planning isn't considering what's actually needed",
                "tier1_categories": "Cognitive",
                "tier2_categories": "Assessing risks",
                "original_text": "Project: Sample Project\nGap: Not thinking carefully about what is needed given risks at this moment",
                "project": "Sample Project"
            },
            {
                "id": "02",
                "gap_text": "Struggling to visualize the problem clearly",
                "other_content": "Title: Lacks clear representation\nContext: Not sure how to represent the complex problem",
                "tier1_categories": "Cognitive",
                "tier2_categories": "Representing problem and solution spaces",
                "original_text": "Project: Another Project\nGap: Struggling to visualize the problem clearly",
                "project": "Another Project"
            }
        ]
        with open(peer2peer_cases, "w", encoding="utf-8") as f:
            json.dump(sample_cases, f, indent=2)
    
    # Check for base template and create if needed
    base_template = templates_dir / "base_template.md"
    if not base_template.exists():
        print(f"Creating base template at {base_template}")
        with open(base_template, "w", encoding="utf-8") as f:
            f.write("""# Personalized Learning Plan

## Understanding Your Regulation Gap

[LLM will insert personalized description of the regulation gap]

## How This Relates to a Similar Case

[LLM will connect the student's situation to a similar case]

## Practice Exercises for This Week

1. [Exercise 1]
2. [Exercise 2]
3. [Exercise 3]

## Reflection Prompts

* [Reflection prompt 1]
* [Reflection prompt 2]
* [Reflection prompt 3]
""")
    
    # Check for risks template and create if needed
    risks_template = templates_dir / "assessing_risks_template.md"
    if not risks_template.exists():
        print(f"Creating risks assessment template at {risks_template}")
        with open(risks_template, "w", encoding="utf-8") as f:
            f.write("""# Personalized Learning Plan: Improving Risk Assessment

## Understanding Your Regulation Gap

You're facing challenges with **assessing risks** in your work. This is a common cognitive gap that affects your ability to identify what's most important to focus on next and how to prioritize your efforts effectively.

Based on the analysis of your work, you tend to [LLM will insert specific patterns observed in the student's approach to risks].

## How This Relates to a Similar Case

[LLM will connect to a similar case example]

## Understanding Risk in Design Research

Risk exists when variables in your causal model represent:
* **Unmet conditions**: Things you know are needed but haven't achieved (e.g., not knowing the root cause of user needs, or creating solutions users don't value)
* **Unknown conditions**: Things you're uncertain about (e.g., being unsure about your assumptions, not knowing if users will value your solution)

The goal of risk assessment is to identify these gaps and prioritize addressing the ones most critical to project success.

## Reflection on Recent Learning

Take time to document what you've learned recently:

* **New Insights**: What new things did you learn last week about your problem, users, designs, etc.?
* **Changed Understanding**: How has your understanding evolved from previous weeks?
* **Validated Assumptions**: Which of your assumptions have you confirmed?
* **Invalidated Assumptions**: Which assumptions have you discovered were incorrect?

## Identifying Gaps in Your Understanding

1. Map your current causal model and identify **specific unknown or unmet conditions**:
   * What root causes are you still unsure about?
   * Which user needs remain unvalidated?
   * What aspects of your solution haven't been tested with users?
   * What connections in your causal model are based on assumptions rather than evidence?

2. For each gap identified, explain **why it's a risk**:
   * How could this unknown/unmet condition affect project outcomes?
   * What might happen if your assumption about this variable is wrong?
   * How might this gap create misalignment between your solution and user needs?

3. **Hint**: Review your research canvas - have you answered all the scaffolding questions for each prompt?

## Prioritizing Risks for Your Next Sprint

1. Identify the **1-2 highest priority risks** to address next:
   * Risk 1: [Description]
   * Risk 2: [Description]

2. For each prioritized risk, explain:
   * Why this risk takes precedence over others
   * What specific research activities will help address this risk
   * What evidence would indicate you've sufficiently addressed it

## Practice Exercises for This Week

1. **Causal Model Risk Map**: Create a visual map of your project's causal model. Highlight variables that represent unknown or unmet conditions, and indicate the level of risk each presents.

2. **Assumption Testing Plan**: For your top 3 risks:
   * List the key assumptions underlying each risk
   * Design a specific test for each assumption
   * Define what evidence would confirm or refute each assumption

3. **Pre-Mortem Analysis**: Imagine your project has failed specifically because of unaddressed risks in your causal model. Write a brief "post-mortem" describing:
   * Which unmet/unknown conditions led to failure
   * What signals you missed that could have highlighted these risks
   * How you could have better assessed and addressed these risks

## Reflection Prompts

* What patterns do you notice in the types of unknown/unmet conditions you tend to overlook?
* How might you integrate risk assessment more systematically into your research process?
* What barriers prevent you from thoroughly assessing risks before moving forward?
* When have you successfully identified and addressed a significant risk? What approach did you use?
""")
    
    print("Setup complete! All necessary files and directories have been created.")

if __name__ == "__main__":
    setup_data_directories() 