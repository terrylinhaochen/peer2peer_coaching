import streamlit as st
import json
import pandas as pd
from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv
import time
import io

# Load environment variables (for local development)
load_dotenv()

# Try alternative import approach with better secrets handling
try:
    from openai import OpenAI
    
    # Handle secrets more gracefully
    api_key = None
    try:
        # Try to get from Streamlit secrets first (for cloud deployment)
        api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError):
        # Fall back to environment variable (for local development)
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    
except ImportError:
    # Fallback to older API
    import openai
    
    # Handle secrets more gracefully for older API
    api_key = None
    try:
        api_key = st.secrets["openai"]["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError):
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()
    
    openai.api_key = api_key
    client = openai

# Initialize session state for tracking UI flow
if 'page' not in st.session_state:
    st.session_state.page = 'input'  # 'input', 'edit', 'results', or 'template'

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

if 'extracted_fields' not in st.session_state:
    st.session_state.extracted_fields = {
        'title': '',
        'gap': '',
        'context': '',
        'plan': '',
        'coach_suggestion': ''
    }

if 'meeting_type' not in st.session_state:
    st.session_state.meeting_type = 'SIG Meeting'

# Add new session state for tracking expanded case
if 'expanded_case' not in st.session_state:
    st.session_state.expanded_case = None

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
        model="gpt-4o-mini",  # Much cheaper than gpt-4
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
        model="gpt-4o-mini",  # Much cheaper than gpt-4
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Add new function to generate "How This Applies To You" section
def generate_application_strategies(student_note, diagnosis, similar_case):
    """Generate personalized application strategies based on the diagnosis and similar case"""
    prompt = f"""Create 3-4 concise, actionable strategies for applying lessons from a similar case to this student's situation.

**Student's Situation:**
{student_note}

**Regulation Gap:** {diagnosis["tier1_categories"]} - {diagnosis["tier2_categories"]}

**Similar Case:** {similar_case["gap_text"]}

**Instructions:**
- Keep each strategy to 1-2 sentences maximum
- Focus on specific, actionable advice
- Use bold headers for strategy names (2-4 words)
- Make it directly applicable to their situation

**Format exactly as:**
**Strategy Name:** Brief, actionable description in 1-2 sentences.

**Another Strategy:** Another brief, actionable description.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Add new function for generating suggested questions with better formatting
def generate_suggested_questions(student_note, diagnosis, similar_case):
    """Generate suggested questions for peer conversations based on regulation gap and similar case"""
    prompt = f"""Generate 6-9 focused questions for peer conversations about this regulation gap.

**Student Assessment:** {student_note}
**Gap Type:** {diagnosis["tier1_categories"]} - {diagnosis["tier2_categories"]}
**Similar Case:** {similar_case["gap_text"]}

**Create questions in these categories:**

1. **Understanding the Gap** (2-3 questions about how the gap manifests)
2. **Learning from Experience** (2-3 questions about what worked/didn't work)  
3. **Applying Solutions** (2-3 questions about implementing changes)

**Guidelines:**
- Keep questions concise and specific
- Focus on regulation skills, not just content
- Include one challenging/probing question per category
- Make them conversation starters, not interviews
- CRITICAL: Each bullet point must be on its own separate line
- Never put multiple questions on the same line

**Format exactly as:**

**Understanding the Gap**

• Question about manifestation?
• Question about patterns?

**Learning from Experience**

• Question about what worked?
• Question about challenges?

**Applying Solutions**

• Question about implementation?
• Question about transfer?

**FORMATTING RULES:**
- Each bullet point on its own line
- Include blank lines between sections
- Start each bullet with • followed by a space
- End each question with a question mark
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7
    )
    
    # Get the raw response
    raw_response = response.choices[0].message.content.strip()
    
    # Enhanced processing to ensure proper bullet point separation
    lines = raw_response.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip completely empty lines during processing
        if not line:
            continue
            
        # Handle section headers (bold text without question marks)
        if line.startswith('**') and line.endswith('**') and '?' not in line:
            # Add spacing before section headers (except first one)
            if formatted_lines:
                formatted_lines.append("")
            formatted_lines.append(line)
            formatted_lines.append("")  # Always add space after header
            
        # Handle lines that contain bullet points
        elif '•' in line:
            # Check if multiple bullet points are crammed on one line
            if line.count('•') > 1:
                # Split by bullet points and process each separately
                parts = line.split('•')
                for i, part in enumerate(parts):
                    if i == 0:  # First part before any bullet
                        continue
                    part = part.strip()
                    if part:
                        # Check if this part contains another question (look for '?')
                        if '?' in part:
                            # Further split by question marks if multiple questions
                            questions = part.split('?')
                            for j, question in enumerate(questions):
                                question = question.strip()
                                if question:  # Not empty
                                    if j < len(questions) - 1:  # Not the last part
                                        formatted_lines.append(f"• {question}?")
                                    elif question != "":  # Last part, add only if not empty
                                        formatted_lines.append(f"• {question}?")
                        else:
                            formatted_lines.append(f"• {part}")
            else:
                # Single bullet point on line
                if line.startswith('•'):
                    formatted_lines.append(line)
                else:
                    # Bullet point not at start, fix it
                    bullet_part = line.split('•', 1)[1].strip()
                    if bullet_part:
                        formatted_lines.append(f"• {bullet_part}")
                        
        # Handle other bullet formats (-, *)
        elif line.startswith(('-', '*')):
            question_text = line.lstrip('-*').strip()
            if question_text:
                formatted_lines.append(f"• {question_text}")
                
        # Handle regular text (non-headers, non-bullets)
        else:
            formatted_lines.append(line)
    
    # Join with line breaks
    result = '\n'.join(formatted_lines)
    
    # Clean up excessive spacing while maintaining section breaks
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    
    # Final check: ensure no bullet points are on the same line
    final_lines = result.split('\n')
    truly_final_lines = []
    
    for line in final_lines:
        if line.count('•') > 1:
            # Still have multiple bullets on one line, fix it
            parts = line.split('•')
            for i, part in enumerate(parts):
                if i == 0:
                    continue
                part = part.strip()
                if part:
                    truly_final_lines.append(f"• {part}")
        else:
            truly_final_lines.append(line)
    
    return '\n'.join(truly_final_lines).strip()

# Add new function for audio transcription
def transcribe_audio(audio_bytes):
    """Transcribe audio using OpenAI's Whisper API"""
    try:
        # Convert audio bytes to a file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"  # Whisper API needs a filename
        
        # Transcribe using Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def transcribe_uploaded_audio(uploaded_file):
    """Transcribe uploaded audio file using OpenAI's Whisper API"""
    try:
        # Read the uploaded file
        audio_bytes = uploaded_file.read()
        
        # Create a file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = uploaded_file.name  # Use original filename
        
        # Transcribe using Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        return transcript
    except Exception as e:
        st.error(f"Error transcribing uploaded audio: {str(e)}")
        return None

def summarize_transcription(transcription, meeting_type="SIG Meeting"):
    """Summarize the transcription using GPT-4 to extract key coaching points based on meeting type"""
    
    if meeting_type == "SIG Meeting":
        prompt = f"""You are helping a coach organize their verbal assessment notes from a SIG (Student-Instructor/Coach) meeting.
        
        Please analyze this transcription of a coaching session between a coach and student and organize it into the following structure:
        
        **Assessment Title:** [Brief title summarizing the main issue discussed]
        **Gap (What needs improvement):** [The main regulation gap or challenge the coach identified in the student]
        **Context:** [The situation or background where this gap was observed by the coach]
        **Plan & Reflect:** [Any plans mentioned or reflection questions the coach suggested]
        **Coach Practice Suggestion:** [Specific practice exercises or activities the coach recommended]
        
        Focus on the coach's perspective and assessment of the student's learning gaps.
        
        Here's the transcription:
        {transcription}
        
        Please extract and organize the key information into the five categories above. If any category isn't clearly mentioned in the transcription, indicate it as "Not specified" or suggest what might be relevant based on context.
        """
    else:  # Peer Conversation
        prompt = f"""You are helping organize notes from a peer-to-peer discussion between students.
        
        Please analyze this transcription of a peer conversation and organize it into the following structure:
        
        **Assessment Title:** [Brief title summarizing the main topic or challenge discussed]
        **Gap (What needs improvement):** [Learning challenges or knowledge gaps identified during the discussion]
        **Context:** [The situation or background being discussed between peers]
        **Plan & Reflect:** [Any plans, solutions, or reflection points that emerged from the discussion]
        **Coach Practice Suggestion:** [Peer recommendations or study strategies suggested]
        
        Focus on collaborative learning insights and peer-identified improvement areas.
        
        Here's the transcription:
        {transcription}
        
        Please extract and organize the key information into the five categories above. If any category isn't clearly mentioned in the transcription, indicate it as "Not specified" or suggest what might be relevant based on context.
        """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Much cheaper than gpt-4
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

def process_transcription_and_analyze():
    """Combined function to transcribe, summarize, analyze, and navigate to results"""
    with st.spinner("Transcribing and summarizing... (this may take ~2 minutes)"):
        # Summarize the transcription
        summary = summarize_transcription(st.session_state.transcription, st.session_state.meeting_type)
        st.session_state.audio_summary = summary
        
        # Parse the summary to extract components
        lines = summary.split('\n')
        extracted_title = ""
        extracted_gap = ""
        extracted_context = ""
        extracted_plan = ""
        extracted_coach_suggestion = ""
        
        for line in lines:
            if "Assessment Title:" in line or "**Assessment Title:**" in line:
                extracted_title = line.split("Assessment Title:")[-1].replace("**", "").strip()
            elif "Gap (What needs improvement):" in line or "**Gap (What needs improvement):**" in line:
                extracted_gap = line.split("Gap (What needs improvement):")[-1].replace("**", "").strip()
            elif "Context:" in line or "**Context:**" in line:
                extracted_context = line.split("Context:")[-1].replace("**", "").strip()
            elif "Plan & Reflect:" in line or "**Plan & Reflect:**" in line:
                extracted_plan = line.split("Plan & Reflect:")[-1].replace("**", "").strip()
            elif "Coach Practice Suggestion:" in line or "**Coach Practice Suggestion:**" in line:
                extracted_coach_suggestion = line.split("Coach Practice Suggestion:")[-1].replace("**", "").strip()
        
        # Store extracted fields for editing
        st.session_state.extracted_fields = {
            'title': extracted_title if extracted_title else "Audio Assessment",
            'gap': extracted_gap,
            'context': extracted_context,
            'plan': extracted_plan,
            'coach_suggestion': extracted_coach_suggestion
        }
        
        # Navigate to edit page
        go_to_edit_page()
        st.rerun()

def transcribe_and_summarize_recording(audio_data):
    """Transcribe recorded audio and go to edit page"""
    with st.spinner("Transcribing audio... (this may take ~2 minutes)"):
        transcription = transcribe_audio(audio_data)
        if transcription:
            st.session_state.transcription = transcription
            st.success("Audio transcribed successfully!")
            # Automatically continue with summarization
            process_transcription_and_analyze()
        else:
            st.error("Failed to transcribe audio. Please try again.")

def transcribe_and_summarize_upload(uploaded_file):
    """Transcribe uploaded audio and go to edit page"""
    with st.spinner("Transcribing uploaded audio... (this may take ~2 minutes)"):
        transcription = transcribe_uploaded_audio(uploaded_file)
        if transcription:
            st.session_state.transcription = transcription
            st.success("Audio transcribed successfully!")
            # Automatically continue with summarization
            process_transcription_and_analyze()
        else:
            st.error("Failed to transcribe audio. Please try again.")

# Navigation functions
def go_to_edit_page():
    st.session_state.page = 'edit'

def go_to_results_page():
    st.session_state.page = 'results'

def go_to_template_page(case_index):
    st.session_state.page = 'template'
    st.session_state.selected_case = st.session_state.similar_cases[case_index]

def go_back_to_results():
    st.session_state.page = 'results'

def go_back_to_input():
    st.session_state.page = 'input'

def go_back_to_edit():
    st.session_state.page = 'edit'

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
        background: linear-gradient(135deg, #e8f7ee 0%, #f0fdf4 100%);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .application-section h4 {
        color: #15803d;
        margin-top: 0;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .application-section ul {
        margin: 0;
        padding-left: 0;
    }
    
    .application-section li {
        margin-bottom: 12px;
        line-height: 1.5;
        list-style: none;
    }
    
    .questions-section {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .questions-section h4 {
        color: #1d4ed8;
        margin-top: 0;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .questions-section ul {
        margin: 0;
        padding-left: 0;
    }
    
    .questions-section li {
        margin-bottom: 8px;
        line-height: 1.4;
        list-style: none;
    }
    
    .case-button {
        width: 100%;
        text-align: left;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: white;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .case-button:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .case-expanded {
        border-color: #3b82f6;
        background-color: #f8fafc;
    }
    
    .stButton button {
        width: 100%;
    }
    
    .stButton > button {
        width: 100%;
        text-align: left;
        padding: 15px;
        margin: 5px 0;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: white;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: normal;
        height: auto;
        min-height: 60px;
    }
    
    .stButton > button:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    /* Specific styling for primary buttons (transcribe button) */
    .stButton > button[kind="primary"] {
        background-color: #dc2626 !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
        font-weight: 600;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Alternative selector for primary buttons */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #dc2626 !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
        font-weight: 600;
    }
    
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Force primary button styling with more specific selectors */
    button.stButton.stButton-primary {
        background-color: #dc2626 !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
    }
    
    /* Even more specific targeting for Streamlit primary buttons */
    button[data-testid="baseButton-primary"] {
        background-color: #dc2626 !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
        font-weight: 600 !important;
    }
    
    button[data-testid="baseButton-primary"]:hover {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Target by button text content for extra specificity */
    button:contains("🎯 Transcribe"), 
    button:contains("🔍 Analyze") {
        background-color: #dc2626 !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
        font-weight: 600 !important;
    }
    
    button:contains("🎯 Transcribe"):hover, 
    button:contains("🔍 Analyze"):hover {
        background-color: #b91c1c !important;
        border-color: #b91c1c !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Override any Streamlit default primary button styling */
    .stButton button[kind="primary"],
    .stButton button[data-testid="baseButton-primary"],
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color: #dc2626 !important;
        background-image: none !important;
        color: white !important;
        border: 1px solid #dc2626 !important;
        font-weight: 600 !important;
    }
    
    .stButton button[kind="primary"]:hover,
    .stButton button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #b91c1c !important;
        background-image: none !important;
        border-color: #b91c1c !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    h3 {
        margin-top: 20px;
    }
    
    .nav-button {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Specific styling for case buttons (non-primary) */
    .stButton > button:not([kind="primary"]) {
        width: 100% !important;
        text-align: left !important;
        padding: 15px 20px !important;
        margin: 8px 0 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background: white !important;
        color: #374151 !important;
        font-weight: 400 !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        height: auto !important;
        min-height: 60px !important;
        max-height: 80px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-sizing: border-box !important;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1) !important;
        background-color: #f8fafc !important;
    }
    
    /* Ensure all case buttons have consistent spacing */
    div[data-testid="stButton"]:has(button:not([kind="primary"])) {
        margin: 4px 0 !important;
        width: 100% !important;
    }
    
    /* Force uniform button styling for case buttons */
    button[key*="case_button_"] {
        width: 100% !important;
        text-align: left !important;
        padding: 15px 20px !important;
        margin: 8px 0 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background: white !important;
        color: #374151 !important;
        font-weight: 400 !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        height: 60px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
    }
    
    button[key*="case_button_"]:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1) !important;
        background-color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# MAIN UI LOGIC
st.title("Novice to Expert Learning System")


# INPUT PAGE
if st.session_state.page == 'input':
    # Meeting type selector
    st.subheader("Select Meeting Type")
    meeting_type = st.radio(
        "What type of audio are you uploading?",
        ["SIG Meeting", "Peer Conversation"],
        index=0 if st.session_state.meeting_type == "SIG Meeting" else 1,
        help="SIG Meeting: Coach-Student discussion | Peer Conversation: Student-Student discussion"
    )
    st.session_state.meeting_type = meeting_type
    
    # Audio upload input section - dynamic header based on meeting type
    if meeting_type == "SIG Meeting":
        st.header("Upload Your SIG Meeting Audio")
        st.write("Upload your coaching assessment audio file (MP3 format). The system will transcribe and organize your coach-student interaction into structured fields for review.")
    else:
        st.header("Upload Your Peer Conversation Audio")
        st.write("Upload your peer discussion audio file (MP3 format). The system will transcribe and organize your student-student conversation into structured fields for review.")
    
    # Show estimated processing time
    st.info("**Estimated processing time:** ~2 minutes for transcription and summarization")
    
    # Sample demo section
    st.markdown("---")
    st.subheader("See a Sample Run")
    st.write("This is a sample recording of a Special Interest Group within the Design Technology and Research class at Northwestern, focusing on dialogue where the coach asks various questions to perceive and identify the regulation gaps students have encountered that week.")
    
    st.write("Experience the complete analysis workflow with real coaching conversation data")
    
    if st.button("Run Sample Analysis", type="primary", use_container_width=True):
            # Load sample audio file
            try:
                with open("sample_sig.m4a", "rb") as f:
                    sample_audio_bytes = f.read()
                
                st.session_state.meeting_type = "SIG Meeting"  # Set appropriate meeting type
                
                # Process the sample audio through the same workflow
                with st.spinner("Processing sample audio... (this may take ~2 minutes)"):
                    # Transcribe the sample audio
                    transcription = transcribe_audio(sample_audio_bytes)
                    if transcription:
                        st.session_state.transcription = transcription
                        
                        # Summarize the transcription
                        summary = summarize_transcription(transcription, "SIG Meeting")
                        st.session_state.audio_summary = summary
                        
                        # Parse the summary to extract components
                        lines = summary.split('\n')
                        extracted_title = ""
                        extracted_gap = ""
                        extracted_context = ""
                        extracted_plan = ""
                        extracted_coach_suggestion = ""
                        
                        for line in lines:
                            if "Assessment Title:" in line or "**Assessment Title:**" in line:
                                extracted_title = line.split("Assessment Title:")[-1].replace("**", "").strip()
                            elif "Gap (What needs improvement):" in line or "**Gap (What needs improvement):**" in line:
                                extracted_gap = line.split("Gap (What needs improvement):")[-1].replace("**", "").strip()
                            elif "Context:" in line or "**Context:**" in line:
                                extracted_context = line.split("Context:")[-1].replace("**", "").strip()
                            elif "Plan & Reflect:" in line or "**Plan & Reflect:**" in line:
                                extracted_plan = line.split("Plan & Reflect:")[-1].replace("**", "").strip()
                            elif "Coach Practice Suggestion:" in line or "**Coach Practice Suggestion:**" in line:
                                extracted_coach_suggestion = line.split("Coach Practice Suggestion:")[-1].replace("**", "").strip()
                        
                        # Store extracted fields for editing
                        st.session_state.extracted_fields = {
                            'title': extracted_title if extracted_title else "Sample Assessment",
                            'gap': extracted_gap,
                            'context': extracted_context,
                            'plan': extracted_plan,
                            'coach_suggestion': extracted_coach_suggestion
                        }
                        
                        # Navigate to edit page
                        go_to_edit_page()
                        st.rerun()
                    else:
                        st.error("Failed to process sample audio. Please try again.")
            except FileNotFoundError:
                st.error("Sample audio file not found. Please ensure sample_sig.m4a is in the project directory.")
            except Exception as e:
                st.error(f"Error processing sample audio: {str(e)}")
    
    st.markdown("---")
    st.subheader("Upload Your Own Audio")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'mpeg', 'mpga', 'webm'],
        help="Upload an audio file containing your coaching assessment"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Display file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Single button for complete workflow
        if st.button("🎯 Transcribe & Summarize Audio", key="transcribe_analyze_upload", type="primary"):
            transcribe_and_summarize_upload(uploaded_file)
    
    # Display transcription and summary if available (for user reference)
    if 'transcription' in st.session_state:
        with st.expander("View Raw Transcription"):
            st.text_area("Raw Transcription:", st.session_state.transcription, height=150, disabled=True)
    
    if 'audio_summary' in st.session_state:
        with st.expander("View AI Summary"):
            st.markdown(st.session_state.audio_summary)

# EDIT PAGE - New page for reviewing and editing transcribed content
elif st.session_state.page == 'edit':
    st.header("Review & Edit Assessment")
    st.write("Review the AI-generated summary below and edit any fields as needed before proceeding to analysis.")
    
    # Back button
    if st.button("← Back to Audio Input", key="back_to_input_from_edit"):
        go_back_to_input()
        st.rerun()
    
    # Show original transcription in expandable section
    if 'transcription' in st.session_state:
        with st.expander("View Original Transcription"):
            st.text_area("Raw Transcription:", st.session_state.transcription, height=100, disabled=True)
    
    # Editable fields
    st.subheader("Assessment Fields")
    
    project_title = st.text_input(
        "Assessment Title:", 
        value=st.session_state.extracted_fields.get('title', ''),
        help="Brief title summarizing the main issue"
    )
    
    gap_text = st.text_area(
        "Gap (What needs improvement):", 
        value=st.session_state.extracted_fields.get('gap', ''),
        height=100,
        help="The main regulation gap or challenge identified"
    )
    
    context = st.text_area(
        "Context:", 
        value=st.session_state.extracted_fields.get('context', ''),
        height=100,
        help="The situation or background where this gap was observed"
    )
    
    plan_reflect = st.text_area(
        "Plan & Reflect:", 
        value=st.session_state.extracted_fields.get('plan', ''),
        height=100,
        help="Any plans mentioned or reflection questions suggested"
    )
    
    coach_suggestion = st.text_area(
        "Coach Practice Suggestion:", 
        value=st.session_state.extracted_fields.get('coach_suggestion', ''),
        height=100,
        help="Specific practice exercises or activities recommended"
    )
    
    # Update session state with edited values
    st.session_state.extracted_fields['title'] = project_title
    st.session_state.extracted_fields['gap'] = gap_text
    st.session_state.extracted_fields['context'] = context
    st.session_state.extracted_fields['plan'] = plan_reflect
    st.session_state.extracted_fields['coach_suggestion'] = coach_suggestion
    
    # Analyze button
    if st.button("🔍 Analyze Regulation Gap", key="analyze_edited", type="primary"):
        with st.spinner("Analyzing regulation gap..."):
            # Combine all fields into student_note
            student_note = f"Title: {project_title}\nGap: {gap_text}\nContext: {context}\nPlan: {plan_reflect}\nCoach Suggestion: {coach_suggestion}"
            st.session_state.student_note = student_note
            st.session_state.project_title = project_title
            
            # Get diagnosis
            st.session_state.diagnosis = diagnose_regulation_gap(student_note)
            
            # Find similar cases
            st.session_state.similar_cases = find_similar_cases(
                st.session_state.diagnosis['tier1_categories'], 
                st.session_state.diagnosis['tier2_categories'], 
                gap_text, 
                context + " " + plan_reflect + " " + coach_suggestion
            )
            
            # Navigate to results page
            go_to_results_page()
            st.rerun()

# RESULTS PAGE
elif st.session_state.page == 'results':
    # Display back button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Edit", key="back_to_edit"):
            go_back_to_edit()
            st.rerun()
    with col2:
        if st.button("🎵 New Audio Input", key="new_audio_input"):
            go_back_to_input()
            st.rerun()
    
    # Show assessment title at the top
    st.header(f"Assessment: {st.session_state.project_title}")
    
    # Display coach feedback text (from extracted fields)
    if st.session_state.extracted_fields.get('coach_suggestion'):
        st.subheader("Coach Feedback")
        st.info(f"💬 **Coach's Practice Suggestion:** {st.session_state.extracted_fields['coach_suggestion']}")
    
    # Display diagnosis
    st.subheader("Regulation Gap Diagnosis")
    st.write(f"**Tier 1 Categories:** {st.session_state.diagnosis['tier1_categories']}")
    st.write(f"**Tier 2 Categories:** {st.session_state.diagnosis['tier2_categories']}")
    st.write(f"**Reasoning:** {st.session_state.diagnosis['reasoning']}")
    
    # Display similar cases with controlled expansion
    st.subheader("Similar Cases")
    for i, case in enumerate(st.session_state.similar_cases):
        # Check if this case should be expanded
        is_expanded = st.session_state.expanded_case == i
        
        # Create the expander with controlled state
        expander_key = f"case_expander_{i}"
        
        # Use columns to create a clickable header that controls expansion
        header_text = f"Case {i+1}: {case['gap_text'][:100]}..."
        
        if st.button(header_text, key=f"case_button_{i}", help="Click to expand/collapse"):
            # Toggle the expanded case
            if st.session_state.expanded_case == i:
                st.session_state.expanded_case = None  # Close if already open
            else:
                st.session_state.expanded_case = i  # Open this case, close others
            st.rerun()
        
        # Show content only if this case is expanded
        if is_expanded:
            st.markdown("---")
            
            # Case details
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
            
            # Add suggested questions section
            st.subheader("Suggested Discussion Questions")
            st.write("Use these questions to guide peer conversations about this regulation gap:")
            
            # Generate questions if not already cached for this case
            if f'questions_{i}' not in st.session_state:
                with st.spinner("Generating discussion questions..."):
                    st.session_state[f'questions_{i}'] = generate_suggested_questions(
                        st.session_state.student_note,
                        st.session_state.diagnosis,
                        case
                    )
            
            # Display the questions with improved formatting
            questions_content = st.session_state[f"questions_{i}"]
            st.markdown(f'<div class="questions-section">{questions_content}</div>', unsafe_allow_html=True)
            
            st.markdown("---")

# TEMPLATE PAGE
elif st.session_state.page == 'template':
    # Display back button
    if st.button("← Back to Results", key="back_to_results"):
        go_back_to_results()
        st.rerun()
    
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
st.sidebar.title("Research")
st.sidebar.info("""
This system is based on peer-to-peer coaching research that achieved 87.5% precision in diagnosing student regulation gaps.

The system diagnoses learning challenges using AI analysis of coaching conversations, matches students with similar regulation gaps for peer learning, and facilitates conversations with research-backed discussion guides.

Our approach advances AI-driven coaching methodologies by combining semantic matching with structured knowledge categorization across three key domains: Cognitive Skills, Metacognitive Skills, and Emotional Regulation.

Research Blog: https://chenterry.com/product/llmcoaching/
""") 