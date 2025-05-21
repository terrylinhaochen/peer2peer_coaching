# Regulation Gap Learning System

A system that diagnoses student regulation gaps and provides personalized practice templates based on similar cases.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure API keys:
   - Copy `.env-template` to `.env`
   - Add your OpenAI API key to the `.env` file

3. Run the Streamlit app:
   ```
   cd peer2peer
   streamlit run app.py
   ```

4. Open the HTML interface (optional):
   - Open `novice2expert.html` in your browser
   - Or access the Streamlit interface directly at http://localhost:8501

## File Structure

- `app.py`: Main Streamlit application
- `data/`: Contains the codebook and case studies
  - `codebook.txt`: Regulation gap categorization guidelines
  - `tiered_weighted_cases.json`: Case studies for similarity matching
- `templates/`: Contains template files for the learning plans
- `novice2expert.html`: HTML wrapper for the Streamlit app (optional)

## Usage

1. Enter a student's CAP note (ID, Gap, Other Content)
2. Click "Diagnose Regulation Gap"
3. View the diagnosed categories and similar cases
4. Generate a personalized template based on a selected case
5. Download the template as a markdown file # peer2peer_coaching
