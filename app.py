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

# Meeting summary function for detailed analysis
def generate_meeting_summary(transcription_text):
    """Generate a comprehensive meeting summary for the expandable analysis section"""
    prompt = f"""You are analyzing a peer-to-peer conversation between students discussing their regulation practices and weekly progress.

Please provide a comprehensive summary of this peer meeting conversation. Focus on:

1. The main topics and themes discussed
2. Each student's challenges and progress with regulation practices
3. Specific strategies and techniques mentioned
4. Coach feedback that was shared and discussed
5. Key insights or breakthroughs mentioned
6. Plans and commitments made for future improvement

Write this as a flowing narrative summary that captures the essence of the conversation and the collaborative learning that took place.

Here's the conversation:
{transcription_text}

Provide a detailed but concise summary that would help someone understand what was accomplished in this peer meeting."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Simple transcription function for demo
def process_peer_audio_for_summary_action(transcription_text):
    """Process peer conversation for Summary and Action Plan output focused on regulation practices"""
    prompt = f"""You are analyzing a peer-to-peer conversation between students discussing their regulation practices and SIG meeting feedback.

Regulation skills refer to the ability to manage beliefs, emotions, and thoughts in a way that is effective for different situations and help you to achieve your long term goals.

Based on the meeting transcript/recording provided, do the following:

1) Write a 2-sentence summary of what was discussed. The purpose of the summary is to help the student see their progress and practices for the week.

2) Create an action plan for the following week using bullet points. For the action plan, it should include:
   - What the student did well in terms of regulation practices that they should keep building on
   - How to incorporate feedback received from their meeting with the coach
   - How to build upon their work they did well for the next week

Here's the conversation:
{transcription_text}

Format your response exactly as:

**Summary:**
[2 sentences about what was discussed and their progress/practices]

**Action Plan:**
• [What they did well in regulation practices to build on]
• [How to incorporate coach feedback]
• [How to build on this week's progress]
• [Additional specific regulation practice steps]
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
    st.markdown("### LLM Enabled Regulation Coaching System")
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
    
    # Show simple instruction
    st.info("👥 **Meet with your SIG mates.**")
    
    # Checklist Questions Card
    st.subheader("Discussion Checklist")
    st.warning("""
📋 **Peer Conversation Guidelines**

Use these questions to guide your peer discussion:

**About Your Work:**
• Describe your deliverable
• Describe how you worked that week leading up to your SIG meeting  
• What do you think you did well in terms of regulation practices?
• How might that have contributed to your deliverable?
• Why did you work that way and apply certain strategies?
• Did you do anything differently compared to previous weeks that made you more effective?

**About Feedback:**
• Describe the feedback you received during your SIG meeting
• How do you think the feedback relates to how you practiced that week?
• Did you receive any positive feedback from the coach yet? What did that look like?
• Overall, how do you feel about your week's progress after the SIG meeting compared to before the SIG meeting?

**About Next Steps:**
• Describe what your next steps will be for the following week
• What is one thing that you did well this week relating to regulation practices that you want to incorporate into next week?
• How will you also incorporate the feedback from the SIG meeting?
• How (if possible) can you build on this week's progress for next week?

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
Student A: So this week for my deliverable, I had to present our user research findings to the team. I've been working on managing my perfectionism because I usually get paralyzed by wanting everything to be perfect before I share it.

Student B: Oh wow, how did that go? I remember you mentioning that was something you wanted to work on after your SIG meeting.

Student A: Actually, it went better than expected! My coach had told me to try the "good enough to get feedback" approach this week. So instead of spending days perfecting the presentation, I set a timer for 3 hours to create a rough version and then shared it for input.

Student B: That's awesome! I've been dealing with similar regulation challenges around procrastination. After my SIG meeting, my coach pointed out that I wait until the last minute because I'm afraid of not meeting my own high standards.

Student A: Yes! That's exactly it. What did you do differently this week?

Student B: I tried breaking my project into smaller chunks and celebrating small wins. Instead of thinking "I need to finish this entire prototype," I told myself "today I just need to complete the wireframes." It helped manage that overwhelming feeling.

Student A: That's smart. My coach also suggested I notice when I'm getting stuck in perfectionist thinking. This week I caught myself three times starting to over-research instead of just starting to write. Each time I reminded myself that feedback is more valuable than perfection.

Student B: How did your team respond when you shared the rough version?

Student A: They actually loved the collaborative approach! They had suggestions that made the final version way better than if I had tried to perfect it alone. My coach was right - the feedback was more valuable than my perfectionism.

Student B: That's amazing progress. For next week, I want to keep practicing that small-wins approach. But I also got feedback that I need to communicate my progress better to my team instead of just working in isolation.

Student A: Yeah, I think for me, I want to keep using that timer method and also practice sharing work-in-progress more often. It felt scary but it worked so much better.
        """
        
        # Process sample transcript with real analysis
        with st.spinner("Analyzing sample peer conversation..."):
            # Process for Summary and Action Plan
            result = process_peer_audio_for_summary_action(sample_peer_transcript)
            
            # Generate detailed meeting summary for expandable section
            meeting_summary = generate_meeting_summary(sample_peer_transcript)
            
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
            
            # Display Summary and Action Plan as stacked sections
            st.markdown("""
            <div class="result-card">
                <h4>📝 Summary</h4>
                <div>{}</div>
            </div>
            """.format('<br>'.join(summary_section)), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="result-card">
                <h4>🎯 Action Plan</h4>
                <div>{}</div>
            </div>
            """.format('<br>'.join(action_section)), unsafe_allow_html=True)
            
            with st.expander("View Full Meeting Summary"):
                st.markdown(meeting_summary)
    
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
                # Mock transcription for demo - regulation practices focused
                mock_transcription = """
Student A: This week I worked on my design research project. I think I did well with managing my perfectionism by setting daily deadlines instead of trying to make everything perfect. My coach suggested this strategy last week.

Student B: That's really good! I can relate to that struggle. This week I focused on my presentation skills, and I practiced managing my anxiety by doing breathing exercises before practice sessions. My coach gave me feedback about how my nervousness was affecting my message clarity.

Student A: Did the breathing exercises help? I'm curious because I get really anxious about presenting too.

Student B: Yes, definitely! When I managed my anxiety better, I could focus on organizing my thoughts clearly. My coach was right that my rushing through slides was actually making things worse, not better.

Student A: That makes sense. My coach also pointed out that my perfectionism was actually slowing me down rather than improving quality. So this week I practiced her suggestion about setting "good enough" standards for drafts.

Student B: How did that work for your research project?

Student A: Much better! I finished my user interviews on time and had energy left to analyze the data properly. I'm getting better at recognizing when my emotions are driving my decisions instead of strategic thinking.

Student B: I want to keep building on the anxiety management techniques next week, and also try my coach's suggestion about rehearsing presentations with a timer to practice staying within time limits.

Student A: Good plan. I want to continue with the "good enough" draft strategy and also work on my coach's feedback about asking better follow-up questions during interviews.
                """
                
                # Process for Summary and Action Plan
                result = process_peer_audio_for_summary_action(mock_transcription)
                
                # Generate detailed meeting summary for expandable section
                meeting_summary = generate_meeting_summary(mock_transcription)
                
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
                
                # Display Summary and Action Plan as stacked sections
                st.markdown("""
                <div class="result-card">
                    <h4>📝 Summary</h4>
                    <div>{}</div>
                </div>
                """.format('<br>'.join(summary_section)), unsafe_allow_html=True)
                
                st.markdown("""
                <div class="result-card">
                    <h4>🎯 Action Plan</h4>
                    <div>{}</div>
                </div>
                """.format('<br>'.join(action_section)), unsafe_allow_html=True)
                
                with st.expander("View Full Meeting Summary"):
                    st.markdown(meeting_summary)