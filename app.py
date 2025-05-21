import streamlit as st
import json
import pandas as pd
from pathlib import Path
import os
# Try alternative import approach
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["openai"]["OPENAI_API_KEY"] if "openai" in st.secrets else os.getenv("OPENAI_API_KEY"))
except ImportError:
    # Fallback to older API
    import openai
    openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"] if "openai" in st.secrets else os.getenv("OPENAI_API_KEY")
    client = openai
import numpy as np
from dotenv import load_dotenv
import time

# Load environment variables (for local development)
load_dotenv()

# Initialize OpenAI client - check both Streamlit secrets and environment variables
api_key = st.secrets["openai"]["OPENAI_API_KEY"] if "openai" in st.secrets else os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize session state for tracking UI flow
if 'page' not in st.session_state:
    st.session_state.page = 'input'  # 'input', 'results', or 'template'

if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None

if 'similar_cases' not in st.session_state:
    st.session_state.similar_cases = None

if 'current_template' not in st.session_state:
    st.session_state.current_template = None

if 'student_note' not in st.session_state:
    st.session_state.student_note = None

if 'selected_case' not in st.session_state:
    st.session_state.selected_case = None

# Load the codebook for gap analysis
def load_codebook():
    """Load the regulation gap codebook from various possible locations"""
    possible_paths = [
        "data/codebook.txt",                    # Root directory
        "peer2peer/data/codebook.txt",          # From root running in peer2peer subfolder
        "../data/codebook.txt",                 # One level up (if running from peer2peer)
        os.path.join(os.path.dirname(__file__), "data/codebook.txt")  # Relative to script location
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                print(f"Successfully loaded codebook from: {path}")
                return f.read()
        except FileNotFoundError:
            continue
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError(
        "Could not find codebook.txt in any of these locations: " + 
        ", ".join(possible_paths) + 
        ". Please ensure the file exists in one of these locations."
    )

# Load case studies 
def load_case_studies():
    """Load the tiered weighted cases from various possible locations"""
    possible_paths = [
        "data/tiered_weighted_cases.json",                    # Root directory
        "peer2peer/data/tiered_weighted_cases.json",          # From root running in peer2peer subfolder
        "../data/tiered_weighted_cases.json",                 # One level up (if running from peer2peer)
        os.path.join(os.path.dirname(__file__), "data/tiered_weighted_cases.json")  # Relative to script location
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                print(f"Successfully loaded case studies from: {path}")
                return json.load(f)
        except FileNotFoundError:
            continue
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError(
        "Could not find tiered_weighted_cases.json in any of these locations: " + 
        ", ".join(possible_paths) + 
        ". Please ensure the file exists in one of these locations."
    )

# Load templates based on gap type
def load_template(gap_type="base"):
    """Load the appropriate template based on gap type"""
    template_mapping = {
        "Assessing risks": "assessing_risks_template.md",
        # Add other templates as they're created
    }
    
    template_file = template_mapping.get(gap_type, "base_template.md")
    possible_paths = [
        f"templates/{template_file}",                    # Current directory
        f"peer2peer/templates/{template_file}",          # From root running in peer2peer subfolder
        f"../templates/{template_file}",                 # One level up (if running from peer2peer)
        os.path.join(os.path.dirname(__file__), f"templates/{template_file}")  # Relative to script
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                print(f"Successfully loaded template from: {path}")
                return f.read()
        except FileNotFoundError:
            continue
    
    # If we get here, we couldn't find the file
    raise FileNotFoundError(
        f"Could not find template file {template_file} in any of these locations: " + 
        ", ".join(possible_paths) + 
        ". Please ensure the file exists in one of these locations."
    )

# Initialize data
CODEBOOK = load_codebook()
case_studies = load_case_studies()

# Pre-compute embeddings for cases (in production, store these)
@st.cache_resource
def get_case_embeddings():
    embeddings = {}
    for case in case_studies:
        case_text = f"{case.get('tier1_categories', '')} {case.get('tier2_categories', '')} {case.get('gap_text', '')} {case.get('other_content', '')}"
        response = client.embeddings.create(
            input=case_text,
            model="text-embedding-ada-002"
        )
        embeddings[case["id"]] = response.data[0].embedding
    return embeddings

# Main app functions
def diagnose_regulation_gap(student_note):
    """Use Claude/GPT to diagnose the regulation gap based on the codebook"""
    prompt = f"""You are analyzing a student's learning regulation gap.
    
    Based on the following codebook:
    {CODEBOOK}
    
    Analyze this student note and identify the primary regulation gaps:
    {student_note}
    
    Remember to focus primarily (80%) on the assessment part of the note when categorizing, as this is the coach's perceived regulation gap. Be careful not to categorize the implications of the regulation gap.
    
    First, provide your step-by-step reasoning, then list the categories that apply.
    
    Provide your response in this format:
    Reasoning: [your step-by-step reasoning]
    Tier 1 Categories: [comma-separated categories]
    Tier 2 Categories: [comma-separated subcategories]
    """
    
    response = client.chat.completions.create(
        model="gpt-4",  # or claude-3-opus-20240229
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2
    )
    
    analysis = response.choices[0].message.content
    
    # Extract tier 1 and tier 2 categories
    tier1 = None
    tier2 = None
    reasoning = None
    
    for line in analysis.split('\n'):
        if line.startswith("Tier 1 Categories:"):
            tier1 = line.replace("Tier 1 Categories:", "").strip()
        elif line.startswith("Tier 2 Categories:"):
            tier2 = line.replace("Tier 2 Categories:", "").strip()
        elif line.startswith("Reasoning:"):
            reasoning = line.replace("Reasoning:", "").strip()
        # Also check for the alternative format from codebook
        elif line.startswith("Categories:"):
            categories = line.replace("Categories:", "").strip().split()
            if len(categories) >= 1:
                if not tier1:  # Only set if not already set
                    tier1 = categories[0]
            if len(categories) >= 2:
                if not tier2:  # Only set if not already set
                    tier2 = " ".join(categories[1:])
    
    return {
        "tier1_categories": tier1,
        "tier2_categories": tier2,
        "reasoning": reasoning,
        "full_analysis": analysis
    }

def find_similar_cases(tier1, tier2, gap_text, other_content="", top_k=3):
    """Find similar cases using embedding similarity and generate application strategies"""
    # Load case embeddings
    case_embeddings = get_case_embeddings()
    
    # Create query embedding
    query_text = f"{tier1} {tier2} {gap_text} {other_content}"
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    query_embedding = response.data[0].embedding
    
    # Calculate similarity with all cases
    similarities = {}
    for case_id, case_embedding in case_embeddings.items():
        similarity = np.dot(query_embedding, case_embedding)
        similarities[case_id] = similarity
    
    # Sort by similarity and return top k
    sorted_cases = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_case_ids = [case_id for case_id, _ in sorted_cases[:top_k]]
    
    # Get the full case data
    similar_cases = []
    for case_id in top_case_ids:
        for case in case_studies:
            if case["id"] == case_id:
                case_with_similarity = case.copy()
                case_with_similarity["similarity_score"] = similarities[case_id]
                similar_cases.append(case_with_similarity)
                break
    
    return similar_cases

def generate_personalized_template(student_note, diagnosis, similar_case):
    """Generate a personalized practice template based on the diagnosis and similar case"""
    # Determine which template to use based on tier2 categories
    template_content = ""
    if "Assessing risks" in diagnosis.get("tier2_categories", ""):
        template_content = load_template("Assessing risks")
    else:
        template_content = load_template("base")
    
    prompt = f"""You are creating a personalized learning plan for a student based on their regulation gap diagnosis.

Student's Note:
{student_note}

Diagnosis:
{diagnosis["full_analysis"]}

Similar Case Study:
ID: {similar_case["id"]}
Gap: {similar_case["gap_text"]}
Other Content: {similar_case["other_content"]}
Tier 1 Categories: {similar_case["tier1_categories"]}
Tier 2 Categories: {similar_case["tier2_categories"]}

Use this template structure to create the personalized plan:
{template_content}

Fill in all the placeholders in the template with content specific to this student's situation and the similar case.
Ensure you replace the [LLM will insert...] sections with actual content.
"""

    response = client.chat.completions.create(
        model="gpt-4",  # or claude-3-opus-20240229
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Add new function to generate "How This Applies To You" section
def generate_application_strategies(student_note, diagnosis, similar_case):
    """Generate personalized application strategies based on the diagnosis and similar case"""
    prompt = f"""Based on this student's regulation gap diagnosis and the similar case, generate 4-5 specific strategies for how they can apply lessons from the similar case to their situation.

Student's Note:
{student_note}

Diagnosis:
{diagnosis["full_analysis"]}

Similar Case Study:
ID: {similar_case["id"]}
Gap: {similar_case["gap_text"]}
Other Content: {similar_case["other_content"]}
Tier 1 Categories: {similar_case["tier1_categories"]}
Tier 2 Categories: {similar_case["tier2_categories"]}

Format your response as a bulleted list with 4-5 strategies. Each strategy should have:
- A bold headline (1-5 words)
- A brief explanation (1-2 sentences)

Example format:
• **Slow down and deepen understanding**
  Before moving on to new work, make sure you've thoroughly understood current topics.

• **Document your current understanding**
  Explicitly capture examples and update your paper draft before moving to new areas.
"""

    response = client.chat.completions.create(
        model="gpt-4",  # or claude-3-opus-20240229
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Navigation functions
def go_to_results_page():
    st.session_state.page = 'results'

def go_to_template_page(case_index):
    st.session_state.page = 'template'
    st.session_state.selected_case = st.session_state.similar_cases[case_index]

def go_back_to_results():
    st.session_state.page = 'results'

def go_back_to_input():
    st.session_state.page = 'input'

# Custom CSS for better layout
st.markdown("""
<style>
    .personalized-template {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .application-section {
        background-color: #e8f7ee;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .stButton button {
        width: 100%;
    }
    
    h3 {
        margin-top: 20px;
    }
    
    .nav-button {
        margin-top: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# MAIN UI LOGIC
st.title("Novice to Expert Learning System")

# INPUT PAGE
if st.session_state.page == 'input':
    # Input section - simplified
    st.header("Enter Coach's Assessment")
    
    project_title = st.text_input("Assessment Title:", "Planning is too rushed")
    
    gap_text = st.text_area("Gap (What needs improvement):", 
        "Not thinking carefully about what is needed given risks at this moment. Pattern matching to how we worked before (to get started and have momentum, yes... but need slowing now)")
    
    context = st.text_area("Context:", 
        "Sprint planning isn't considering what's actually needed at this moment")
    
    plan_reflect = st.text_area("Plan & Reflect:", 
        "[self-work] When planning, take time to assess what's needed NOW rather than following the same pattern from before\n[reflect] What signals would help you recognize when to slow down vs. speed up?")

    # Button to diagnose
    if st.button("Diagnose Regulation Gap"):
        with st.spinner("Analyzing regulation gap..."):
            # Combine all fields into student_note
            student_note = f"Title: {project_title}\nGap: {gap_text}\nContext: {context}\nPlan: {plan_reflect}"
            st.session_state.student_note = student_note
            st.session_state.project_title = project_title
            
            # Get diagnosis
            st.session_state.diagnosis = diagnose_regulation_gap(student_note)
            
            # Find similar cases
            st.session_state.similar_cases = find_similar_cases(
                st.session_state.diagnosis['tier1_categories'], 
                st.session_state.diagnosis['tier2_categories'], 
                gap_text, 
                context + " " + plan_reflect
            )
            
            # Navigate to results page
            go_to_results_page()
            st.experimental_rerun()

# RESULTS PAGE
elif st.session_state.page == 'results':
    # Display back button
    if st.button("← Back to Input", key="back_to_input"):
        go_back_to_input()
        st.experimental_rerun()
    
    # Show assessment title at the top
    st.header(f"Assessment: {st.session_state.project_title}")
    
    # Display diagnosis
    st.subheader("Regulation Gap Diagnosis")
    st.write(f"**Tier 1 Categories:** {st.session_state.diagnosis['tier1_categories']}")
    st.write(f"**Tier 2 Categories:** {st.session_state.diagnosis['tier2_categories']}")
    st.write(f"**Reasoning:** {st.session_state.diagnosis['reasoning']}")
    
    # Display similar cases
    st.subheader("Similar Cases")
    for i, case in enumerate(st.session_state.similar_cases):
        with st.expander(f"Case {i+1}: {case['gap_text'][:100]}..."):
            st.write(f"**Project:** {case.get('project', 'N/A')}")
            st.write(f"**Gap:** {case.get('gap_text', 'N/A')}")
            st.write(f"**Tier 1 Categories:** {case.get('tier1_categories', 'N/A')}")
            st.write(f"**Tier 2 Categories:** {case.get('tier2_categories', 'N/A')}")
            st.write(f"**Other Content:** {case.get('other_content', 'N/A')}")
            st.write(f"**Similarity Score:** {case.get('similarity_score', 0):.4f}")
            
            # Add "How This Applies To You" section
            application_strategies = generate_application_strategies(
                st.session_state.student_note, 
                st.session_state.diagnosis, 
                case
            )
            st.subheader("How This Applies To You")
            st.markdown(f'<div class="application-section">{application_strategies}</div>', unsafe_allow_html=True)
            
            # Button to generate personalized template - now navigates to template page
            if st.button(f"Generate Personalized Template", key=f"template_{i}"):
                # Set the selected case before switching pages
                st.session_state.selected_case = case
                # Reset any previous template
                st.session_state.current_template = None
                go_to_template_page(i)
                st.experimental_rerun()

# TEMPLATE PAGE
elif st.session_state.page == 'template':
    # Display back button
    if st.button("← Back to Results", key="back_to_results"):
        go_back_to_results()
        st.experimental_rerun()
    
    # Get actual project name from the selected case
    project_name = st.session_state.selected_case.get('project', st.session_state.project_title)
    
    # Show case information as header
    st.header(f"Project: {project_name}")
    st.subheader(f"Personalized Template Based on Similar Case")
    st.write(f"**Regulation Gap:** {st.session_state.selected_case['gap_text']}")
    
    # Generate template if needed
    if st.session_state.current_template is None:
        with st.spinner("Generating personalized practice template..."):
            st.session_state.current_template = generate_personalized_template(
                st.session_state.student_note, 
                st.session_state.diagnosis, 
                st.session_state.selected_case
            )
    
    # Option to download template at the top with correct project name
    st.download_button(
        "Download Complete Template", 
        st.session_state.current_template, 
        file_name=f"practice_template_{project_name.replace(' ', '_')}.md",
        mime="text/markdown"
    )
    
    # Parse template into sections
    template_sections = []
    current_section = {"title": "Introduction", "content": ""}
    
    for line in st.session_state.current_template.split('\n'):
        if line.startswith('## '):
            # Save the previous section
            if current_section["content"]:
                template_sections.append(current_section)
            
            # Start a new section
            section_title = line[3:].strip()
            current_section = {"title": section_title, "content": line + "\n"}
        else:
            # Add to current section
            current_section["content"] += line + "\n"
    
    # Add the last section
    if current_section["content"]:
        template_sections.append(current_section)
    
    # Create a mapping of section titles to input labels
    section_input_mapping = {
        "Understanding Your Regulation Gap": "Your understanding of the gap",
        "How This Relates to a Similar Case": "Your thoughts on the similar case",
        "Understanding Risk in Design Research": "Your understanding of risk",
        "Reflection on Recent Learning": "Your recent learning reflections",
        "Identifying Gaps in Your Understanding": "Your identified gaps",
        "Prioritizing Risks for Your Next Sprint": "Your priority risks",
        "Practice Exercises for This Week": "Your exercise results",
        "Reflection Prompts": "Your reflections"
    }
    
    # Create a container for save button success message
    save_container = st.empty()
    
    # Display each section with its input box
    responses = {}
    
    for i, section in enumerate(template_sections):
        st.markdown("---")
        cols = st.columns([3, 2])
        
        with cols[0]:
            st.markdown(f'<div class="personalized-template">{section["content"]}</div>', unsafe_allow_html=True)
        
        with cols[1]:
            # Get appropriate label for the input box
            input_label = section_input_mapping.get(section["title"], f"Your response to {section['title']}")
            
            # Get a unique key for this text area
            section_key = f"input_{i}_{section['title'].replace(' ', '_')}"
            
            # Create text area with appropriate height based on content length
            content_length = len(section["content"])
            height = min(max(100, content_length // 5), 300)  # Adjust height based on content
            
            responses[section["title"]] = st.text_area(input_label, key=section_key, height=height)
    
    # Save button at the bottom
    if st.button("Save All Responses", key="save_all_responses"):
        # Here you would save responses to a database in a real application
        save_container.success("All responses saved successfully!")

# Sidebar content
st.sidebar.title("About")
st.sidebar.info(
    "This tool helps diagnose student regulation gaps and provides "
    "personalized practice templates based on similar cases."
)
st.sidebar.title("Settings")
st.sidebar.info(
    "This is a prototype. In production, you would want to:"
    "\n- Store embeddings in a vector database"
    "\n- Add user authentication"
    "\n- Implement proper error handling"
    "\n- Add case feedback mechanisms"
) 