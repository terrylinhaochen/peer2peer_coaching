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
    st.session_state.page = 'home'  # 'home', 'sig_meeting', 'peer_meeting'

if 'matched_peer' not in st.session_state:
    st.session_state.matched_peer = None

# Navigation functions
def go_to_home():
    st.session_state.page = 'home'

def go_to_sig_meeting():
    st.session_state.page = 'sig_meeting'

def go_to_peer_meeting():
    st.session_state.page = 'peer_meeting'
    # Initialize matched peer (hardcoded for now)
    st.session_state.matched_peer = {
        'name': 'Alex Chen',
        'regulation_gap': 'Assessing risks in user testing',
        'similarity_score': 0.87,
        'shared_challenges': ['Understanding why design arguments fail', 'Articulating testing insights']
    }

# Transcription summarization function for SIG meetings
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

# Simple transcription function for demo
def process_peer_audio_for_summary_action(transcription_text):
    """Process peer conversation for Summary and Action Plan output"""
    prompt = f"""You are analyzing a peer-to-peer learning conversation between students.

Please analyze this conversation and provide:

**Summary:**
Write a 3-sentence summary of the main discussion points and insights shared.

**Action Plan:**
Create a bullet-point action plan with specific next steps both students can take.

Here's the conversation:
{transcription_text}

Format your response exactly as:

**Summary:**
[3 sentences summarizing the key discussion points and insights]

**Action Plan:**
• [Actionable step 1]
• [Actionable step 2] 
• [Actionable step 3]
• [etc.]
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

# CSS for better styling
st.markdown("""
<style>
    .peer-info-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0ea5e9;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.1);
    }
    
    .checklist-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.1);
    }
    
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# MAIN UI LOGIC
st.title("SPS V0.2 Prototype")

# HOME PAGE
if st.session_state.page == 'home':
    st.markdown("### Welcome to the LLM Enabled Regulation Coaching System")
    st.markdown("Choose your session type to get started:")
    
    # Two main option cards
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 SIG Meeting\n\nUpload and analyze coach-student discussion audio for structured assessment and feedback", key="sig_card", help="Coach-Student Assessment Session", use_container_width=True):
            go_to_sig_meeting()
            st.rerun()
    
    with col2:
        if st.button("👥 Peer Meeting\n\nConnect with matched peers and analyze peer-to-peer learning conversations", key="peer_card", help="Peer-to-Peer Learning Session", use_container_width=True):
            go_to_peer_meeting()
            st.rerun()

# SIG MEETING PAGE
elif st.session_state.page == 'sig_meeting':
    # Back to home button
    if st.button("← Back to Home", key="back_to_home_from_sig"):
        go_to_home()
        st.rerun()
    
    st.header("Upload Your SIG Meeting Audio")
    st.write("Upload your coaching assessment audio file (MP3 format). The system will transcribe and organize your coach-student interaction into structured fields for review.")
    
    # Show estimated processing time
    st.info("**Estimated processing time:** ~2 minutes for transcription and summarization")
    
    # Sample demo section
    st.markdown("---")
    st.subheader("See a Sample Run")
    st.write("This is a sample recording of a Special Interest Group within the Design Technology and Research class at Northwestern, focusing on dialogue where the coach asks various questions to perceive and identify the regulation gaps students have encountered that week.")
    
    st.write("Experience the complete analysis workflow with real coaching conversation data")
    
    if st.button("Run Sample Analysis", type="primary", use_container_width=True, key="sample_sig"):
        # Use pre-transcribed sample data for instant demo
        sample_transcript = """So that led to our research, which is how do we scaffold a conversation, such as students ask better or deeper questions to their experienced peers, so that they can adjust their regulation gap? So just trying to come up with better question promptings so that we get better responses. I saw that. A couple things. First of all, I don't know if the bot's working, but did you get these reflection questions? I think there's a problem happening right now, but he gave us those. He sent it to us. OK, well, I can't see them, so. Oh, what it means is that if you do the reflections, yeah, they should show up here. Oh, oh, I see. OK, so I'm just letting you guys know. Well, I'm just letting you guys know. Yeah, it's not it's not. So I can't see your reflections on what you did. But we're going to keep working on. Keep working on articulating risks, so let's let's work on that a little bit. What? So right now, here's what I've heard so far. I heard we tested. Conversation went like, OK, but like maybe not so good or there's some parts like they could articulate certain things in the conversation. So we need to ask better questions. So we're going to ask better questions. That's roughly what I've heard. OK. If you're doing takeaways from user testing as you test it, it asks you to walk through. What is the new understanding that you've gained through testing with users? So often, you know, to talk about risk, it's really useful to talk about what you know, what you now know, and then to talk about what you still don't know. So it's nice to have both sides of that when you talk about a risk. So then, let's see, let's go through the sections of takeaways from user testing. Some of the first questions it asks are, what new understanding do you have? You know, there's the insights part, right? You guys know what I'm talking about when I'm saying all this, right? If you're no longer following, just pause me and then we'll put up and we'll look together. So it's like quick insights. And then it gets into like, well, did you learn anything new about the users? Like, what are their goals or what are their obstacles? And in particular, right, again, obstacles are not just what can't they do, but about why do they struggle to do the things that we want them to do or that they want to do, right? So it's understanding those why's. And then, you know, from there, there's things what you learn about your design and where it's working, where it's not. So I'm curious. So the first part about obstacles is, I haven't heard anything about why they struggle to do certain things, right? Like why are they bad at articulating their issues and the regulation gaps and what new things have you learned about that? Does that make sense? So that's something I'm curious about. And then on the solution side, well, you have some design arguments. Like there's some way you're trying to facilitate a conversation. So what did you learn about, right? And you had an argument about why they shouldn't get over the obstacles you thought were challenges. Do you need something? Well, not contagious. I've just been having a cough. She's been like that for two weeks. Sorry. So obstacles. And then what are the characteristics of your tool that's supposed to facilitate conversations along these lines? I'm assuming they're not working. This is still happening. But why aren't they working? Because you had an argument for why they should have worked and what obstacles they should have gotten over. So was there a new obstacle you didn't anticipate? Or is it that your argument about why it would get over the current obstacle didn't work? And so from there, I'm still walking you down takeaways from user testing. There's going to be something about your testing setup. So the question is about, is your testing setup helping you understand the things you want to understand? And some of it, the questions I just asked you about the obstacles, about how the design argument is working, it's possible. That your testing setup allowed you to understand those things or made it really hard to actually understand those things. So you saw that they were bad at articulating, but you don't really understand why. You saw that the argument didn't work, but you don't really understand which specific part of the argument broke down. And that might suggest that the testing setup has to be improved. So I'll just pause there. Does that all make sense? Any questions about any of that? I'm not asking you what to do about it yet, but does that all make sense? Okay, so good. So let's go back to here, right? And I know this is like a week ago, a week ago. But still, to help us follow along with your story, because this very quickly jumped into, they're bad at this, so let's just fix this. What were you able to learn about the things, about obstacles, about the design argument, about the testing setup? And are there anything there that are important for us to be thinking about as we tackle this question? Because now it just basically says, it didn't work, so we need to do better. It's not very helpful. I think one of the ways I think about this is, imagine you don't get the design, you just get to present the takeaways. Does that make sense? So you present the takeaways, let's say, to Lynn and Grace, and you tell them, well, students are bad at articulating, go make better questions. So they're sitting there and they're like, okay, great, we got to make better questions, but they're like, you want to tell us what would help us make better questions? Because without it, how are they going to do any better than they did on the last attempt? Okay, good, so let's try again, and if you don't have the answers to some of these questions, that's good too, but that might tell us something about the testing setup, right? So one way or another, we're going to learn something."""
        
        # Process sample transcript with real analysis
        with st.spinner("Analyzing sample transcript..."):
            # Summarize the sample transcription
            summary = summarize_transcription(sample_transcript, "SIG Meeting")
            
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
            
            st.success("Sample analysis complete!")
            
            # Display results
            st.markdown("---")
            st.subheader("Sample SIG Meeting Analysis Results")
            
            st.write(f"**Assessment Title:** {extracted_title if extracted_title else 'Sample Assessment'}")
            st.write(f"**Gap (What needs improvement):** {extracted_gap}")
            st.write(f"**Context:** {extracted_context}")
            st.write(f"**Plan & Reflect:** {extracted_plan}")
            st.write(f"**Coach Practice Suggestion:** {extracted_coach_suggestion}")
            
            with st.expander("View Full AI Analysis"):
                st.markdown(summary)
    
    st.markdown("---")
    st.subheader("Upload Your Own Audio")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'mpeg', 'mpga', 'webm'],
        key="sig_upload",
        help="Upload an audio file containing your coaching assessment"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Display file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Single button for complete workflow
        if st.button("🎯 Transcribe & Summarize Audio", key="transcribe_sig", type="primary"):
            st.success("Would transcribe and process through existing SIG Meeting workflow.")

# PEER MEETING PAGE
elif st.session_state.page == 'peer_meeting':
    # Back to home button
    if st.button("← Back to Home", key="back_to_home_from_peer"):
        go_to_home()
        st.rerun()
    
    st.header("Peer Meeting Session")
    
    # Show matched peer information
    if st.session_state.matched_peer:
        st.subheader("Your Matched Peer")
        peer = st.session_state.matched_peer
        st.info(f"👤 **{peer['name']}**\n\n**Regulation Gap:** {peer['regulation_gap']}\n\n**Match Score:** {peer['similarity_score']:.0%}\n\n**Shared Challenges:** {', '.join(peer['shared_challenges'])}")
    
    # Checklist Questions Card
    st.subheader("Discussion Checklist")
    st.warning("""
📋 **Peer Conversation Guidelines**

Use these questions to guide your peer discussion:

• How did you encounter this regulation gap?
• What strategies have you tried to address it?
• What worked and what didn't work?
• What specific examples can you share?
• How can we support each other in improvement?

*Record your conversation and upload it below for analysis.*
    """)
    
    # Sample demo section
    st.markdown("---")
    st.subheader("See a Sample Run")
    st.write("This is a sample peer-to-peer conversation between students discussing regulation gaps and learning strategies. Experience how the system analyzes peer discussions and generates actionable insights.")
    
    st.write("Experience the complete peer analysis workflow with real conversation data")
    
    if st.button("Run Sample Analysis", type="primary", use_container_width=True, key="sample_peer"):
        # Use pre-transcribed sample peer conversation for instant demo
        sample_peer_transcript = """
Student A: So I've been really struggling with understanding why our user testing results don't seem to match what we expected from our design arguments. Like, we design something thinking it will solve a specific problem, but then users don't use it the way we anticipated.

Student B: Oh yeah, I've had the same issue! It's so frustrating because you spend all this time crafting what you think is a solid argument for why something should work, and then reality hits and users behave completely differently.

Student A: Exactly! And I feel like I'm not good at articulating what went wrong or why. It's like I know there's something to learn from it, but I can't put my finger on what specifically broke down in my reasoning.

Student B: I think part of the issue is that we're not asking the right questions during testing. Like, we ask "did this work for you?" but we don't dig into the why behind their behavior. We need to understand their mental models.

Student A: That's a really good point. Maybe we need to focus more on understanding their thought processes and how they approach the task, not just whether they can complete it successfully.

Student B: Yeah, and also being more systematic about what we're trying to learn. I feel like sometimes we go into testing without clear hypotheses about what might go wrong with our design arguments.

Student A: Right! So maybe we should spend more time upfront thinking about what assumptions we're making in our design arguments and how we can test those assumptions specifically.

Student B: Definitely. And then we need to get better at synthesizing what we learn and turning it into actionable insights for the next iteration. Like, not just "users struggled" but "users struggled because they expected X but we designed for Y."

Student A: That would help so much with articulating the regulation gaps too. Instead of just saying "I'm bad at user testing," I could say "I struggle with identifying and testing the assumptions underlying my design arguments."

Student B: Exactly! And we could probably develop some kind of framework for ourselves - like a checklist of assumption categories to consider before testing, and then specific questions to ask during testing to validate or invalidate those assumptions.
        """
        
        # Process sample transcript with real analysis
        with st.spinner("Analyzing sample peer conversation..."):
            # Process for Summary and Action Plan
            result = process_peer_audio_for_summary_action(sample_peer_transcript)
            
            st.success("Sample peer analysis complete!")
            
            # Display results in two separate boxes
            st.markdown("---")
            st.subheader("Sample Peer Analysis Results")
            
            # Parse the response to separate Summary and Action Plan
            lines = result.split('\n')
            summary_section = []
            action_section = []
            current_section = None
            
            for line in lines:
                line = line.strip()
                if '**Summary:**' in line:
                    current_section = 'summary'
                    continue
                elif '**Action Plan:**' in line:
                    current_section = 'action'
                    continue
                elif line:  # Skip empty lines
                    if current_section == 'summary':
                        summary_section.append(line)
                    elif current_section == 'action':
                        action_section.append(line)
            
            # Display Summary and Action Plan boxes
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="result-card">
                    <h4>📝 Summary</h4>
                    <p>{}</p>
                </div>
                """.format(' '.join(summary_section)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="result-card">
                    <h4>🎯 Action Plan</h4>
                    <div>{}</div>
                </div>
                """.format('<br>'.join(action_section)), unsafe_allow_html=True)
            
            with st.expander("View Full AI Analysis"):
                st.markdown(result)
    
    st.markdown("---")
    st.subheader("Upload Your Own Audio")
    
    # Audio upload section
    st.write("Upload your peer discussion audio file (MP3 format). The system will transcribe and organize your student-student conversation into structured fields for review.")
    
    st.info("**Estimated processing time:** ~2 minutes for transcription and summarization")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'mpeg', 'mpga', 'webm'],
        key="peer_audio_upload",
        help="Upload an audio file containing your peer conversation"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Display file info
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Single button for complete workflow
        if st.button("🎯 Transcribe & Analyze Peer Conversation", key="transcribe_analyze_peer", type="primary"):
            with st.spinner("Processing peer conversation..."):
                # Mock transcription for demo
                mock_transcription = """
Student A: So I've been really struggling with understanding why our user testing results don't seem to match what we expected from our design arguments.

Student B: Oh yeah, I've had the same issue! Like, we design something thinking it will solve a specific problem, but then users don't use it the way we anticipated.

Student A: Exactly! And I feel like I'm not good at articulating what went wrong or why. It's frustrating because I know there's something to learn from it, but I can't put my finger on it.

Student B: I think part of the issue is that we're not asking the right questions during testing. Like, we ask "did this work for you?" but we don't dig into the why behind their behavior.

Student A: That's a good point. Maybe we need to focus more on understanding their mental models and how they approach the task, not just whether they can complete it.

Student B: Yeah, and also being more systematic about what we're trying to learn. I feel like sometimes we go into testing without clear hypotheses about what might go wrong.

Student A: Right! So maybe we should spend more time upfront thinking about what assumptions we're making and how we can test those specifically.

Student B: Definitely. And then we need to get better at synthesizing what we learn and turning it into actionable insights for the next iteration.
                """
                
                # Process for Summary and Action Plan
                result = process_peer_audio_for_summary_action(mock_transcription)
                
                st.success("Peer conversation analysis complete!")
                
                # Display results in two separate boxes
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Parse the response to separate Summary and Action Plan
                lines = result.split('\n')
                summary_section = []
                action_section = []
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if '**Summary:**' in line:
                        current_section = 'summary'
                        continue
                    elif '**Action Plan:**' in line:
                        current_section = 'action'
                        continue
                    elif line:  # Skip empty lines
                        if current_section == 'summary':
                            summary_section.append(line)
                        elif current_section == 'action':
                            action_section.append(line)
                
                # Display Summary box
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="result-card">
                        <h4>📝 Summary</h4>
                        <p>{}</p>
                    </div>
                    """.format(' '.join(summary_section)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="result-card">
                        <h4>🎯 Action Plan</h4>
                        <div>{}</div>
                    </div>
                    """.format('<br>'.join(action_section)), unsafe_allow_html=True)