import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import altair as alt
import numpy as np
import os # Import os module to check for file existence
import time
# --- Constants and Config ---
ADMIN_PASSWORD = os.environ.get('EDUPULSE_ADMIN_PASS', 'adminpass')

# --- Streamlit Page Configuration ---

st.set_page_config(
    page_title="Edupulse WUA",
    layout="wide",  # Use a wide layout for better visualization
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Custom styling: deep bottle green background and yellow accents ---
st.markdown("""
<style>
/* Main background gradient */
html, body, [data-baseweb="appContainer"], .stApp {
    background: linear-gradient(180deg, #1B4D3E 0%, #0F3528 50%, #1B4D3E 100%) !important;
}

/* Force text to be white everywhere */
h1, h2, h3, p, span, div, label, li, input {
    color: #ffffff !important;
}

/* Ensure header is transparent */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* All Buttons (including the Login Form submit button) */
button {
    background-color: #2d7a6b !important;
    border: 2px solid #fff59d !important;
    border-radius: 5px !important;
}
button:hover {
    background-color: #fff59d !important;
}
button:hover p, button:hover span, button:hover div {
    color: #1B4D3E !important; 
}

/* Textboxes, Password fields, and Selectbox backgrounds */
[data-baseweb="select"] > div, 
[data-baseweb="input"] > div,
input {
    background-color: #0F3528 !important;
    border: 1px solid #2d7a6b !important;
}

/* The dropdown menu options container */
ul[data-baseweb="menu"], ul[role="listbox"], div[data-baseweb="popover"] > div {
    background-color: #0F3528 !important;
}
li[role="option"] {
    background-color: transparent !important;
}

/* Fix chatbot background visibility (dark green) */
[data-testid="stChatMessage"] {
    background-color: rgba(31, 77, 62, 0.8) !important;
    border: 1px solid #2d7a6b !important;
    border-radius: 10px;
    padding: 10px;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0F3528 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"] p {
    color: #fff59d !important;
    font-weight: bold;
}

/* Form background container */
[data-testid="stForm"] {
    background: rgba(15, 53, 40, 0.8) !important;
    border: 1px solid #2d7a6b !important;
    border-radius: 10px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

def safe_rerun():
    """Try to rerun the Streamlit script; fallback to mutating query params if rerun API isn't available."""
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.experimental_set_query_params(_rerun=str(time.time()))
        except Exception:
            st.stop()

def logout():
    for k in ['logged_in', 'role', 'chat_history']:
        if k in st.session_state:
            del st.session_state[k]
    safe_rerun()

# Add logo and title
col1, col2, col3 = st.columns([0.8, 2, 2.2])

if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = 'Home'

with col1:
    logo_path = os.path.join(os.path.dirname(__file__), 'WUA logo.png')
    st.image(logo_path, width=100)

with col2:
    st.title("🎓 Edupulse")
    st.markdown("#### Student Support & Dropout Prevention Platform")

with col3:
    st.markdown("<br>", unsafe_allow_html=True) # push down slightly so it aligns with title
    # Custom CSS for these specific tiny navigation buttons so they fit
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-of-type(3) .stButton>button {
        padding: 0.2rem 0.5rem !important;
        font-size: 0.8rem !important;
        height: auto !important;
        min-height: 30px !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns(5)
    with nav_c1:
        if st.button("Home"): st.session_state.nav_selection = "Home"
    with nav_c2:
        if st.button("About"): st.session_state.nav_selection = "About"
    with nav_c3:
        if st.button("Contact"): st.session_state.nav_selection = "Contact"
    with nav_c4:
        if st.button("Lang"): st.session_state.nav_selection = "Language"
    with nav_c5:
        logged_in = st.session_state.get('logged_in', False)
        if st.button("Logout" if logged_in else "Login"):
            if logged_in:
                logout()
            else:
                st.session_state.nav_selection = "Login"
                safe_rerun()

st.markdown("""
---
This dashboard offers an integrated platform for student support, featuring:
- **Chatbot**: An AI assistant for common student queries.
- **Dropout Predictor**: A tool to assess student dropout risk based on key metrics.
- **Insights**: Visualizations of student data and dropout risk factors.
""")

if st.session_state.nav_selection == "About":
    st.header("About Edupulse")
    st.write("Edupulse is a Student Support & Dropout Prevention Platform designed to improve student retention and success. It assists students with quick inquiries through our AI Chatbot and helps administration seamlessly monitor student engagement and risk factors.")
    st.stop()
elif st.session_state.nav_selection == "Contact":
    st.header("Contact Us")
    st.markdown("### Developer")
    st.write("**Name:** Chara RN Mabeza")
    st.write("**Title:** Developer")
    st.write("**Email:** charamabeza@gmail.com")
    st.write("**Phone:** 0773512390")
    st.stop()
elif st.session_state.nav_selection == "Language":
    st.header("Language Settings")
    st.write("Current Language: English")
    st.info("Additional language options are under development.")
    st.stop()
elif st.session_state.nav_selection == "Login":
    st.header("Welcome to Edupulse")
    st.markdown("Please log in to continue.")

    with st.form('nav_login_form'):
        role_choice = st.selectbox('I am a/an', ['Student', 'Admin'])
        password = ''
        if role_choice == 'Admin':
            password = st.text_input('Admin password', type='password')
        submit = st.form_submit_button('Login')

    if submit:
        if role_choice == 'Admin':
            if password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.role = 'Admin'
                st.session_state.nav_selection = "Home"
                st.success('Logged in as Admin')
                safe_rerun()
            else:
                st.error('Invalid admin password')
        else:
            st.session_state.logged_in = True
            st.session_state.role = 'Student'
            st.session_state.nav_selection = "Home"
            st.success('Logged in as Student')
            safe_rerun()
    st.stop()

# --- Chatbot Components (Cached to run once) ---
@st.cache_resource
def load_chatbot_components():
    """
    Loading and training the necessary components for the chatbot (data, vectorizer, model).
    This function is cached using st.cache_resource to run only once when the app starts.
    """
    # 1. Data Preparation: Synthetic dataset of student queries and intents
    intents = ['deadlines', 'registration', 'library', 'fees', 'exams', 'support', 'timetable']
    synthetic_data = {
        'deadlines': [
            'When is the deadline for assignment submission?', 'What are the upcoming deadlines for coursework?',
            'Can you tell me the deadline for my essay?', 'Is there a deadline for course registration?',
            'When do I need to submit my final project?', 'What is the last date to drop a course?',
            'When are the application deadlines?', 'Can I get an extension on my deadline?',
            'When is the payment deadline?', 'Are there any deadlines approaching for scholarships?'
        ],
        'registration': [
            'How do I register for new courses?', 'What is the procedure for course registration?',
            'When does course registration open?', 'I need help with my registration.',
            'Can I change my registered courses?', 'What documents are required for registration?',
            'Is late registration possible?', 'How to add/drop courses?',
            'What are the steps for university registration?', 'I forgot my registration password.'
        ],
        'library': [
            'What are the library opening hours?', 'How can I borrow a book from the library?',
            'Do you have resources for research in the library?', 'How to access online library databases?',
            'Can I reserve a study room in the library?', 'Where is the main library located?',
            'How many books can I check out from the library?', 'Is there a quiet study area in the library?',
            'Can I print documents at the library?', 'I lost my library card.'
        ],
        'fees': [
            'What are the tuition fees for this semester?', 'How can I pay my university fees?',
            'What is included in the fees?', 'Are there any scholarships to help with fees?',
            'When is the fee payment due?', 'Can I pay fees in installments?',
            'What happens if I miss the fee payment deadline?', 'Do international students pay different fees?',
            'How much are the lab fees?', 'Where can I find information about fee refunds?'
        ],
        'exams': [
            'When are the final exams scheduled?', 'What is the exam timetable?',
            'Can I get my exam results?', 'How to apply for a re-sit exam?',
            'What is the policy for deferred exams?', 'Where can I find past exam papers?',
            'Are there any study guides for the exams?', 'How long are the exams?',
            'What items are allowed in the exam hall?', 'I have a conflict with my exam schedule.'
        ],
        'support': [
            'I need academic support.', 'Where can I find mental health support?',
            'How to contact student support services?', 'Do you offer career counseling?',
            'I need help with my visa application.', 'Who can I talk to about personal issues?',
            'Is there a disability support service?', 'How to get technical support?',
            'Can I get help with my resume?', 'I need advice on my course choices.'
        ],
        'timetable': [
            'Where can I find my class timetable?', 'What is my schedule for next week?',
            'Has the timetable been updated?', 'Can I get a personalized timetable?',
            'My timetable has a clash.', 'How to access the online timetable?',
            'When does the new semester timetable come out?', 'I need to change my timetable.',
            'What time does my lecture start?', 'Can you show me the timetable for Computer Science?'
        ],
        'greeting': [
            'Hello', 'Hi', 'Hey', 'Good morning', 'Good afternoon', 'Greetings', 'Hi there', 'Hey chatbot', 'Yo'
        ],
        'thanks': [
            'Thank you', 'Thanks', 'Appreciate it', 'Thanks a lot', 'Thank you so much', 'Awesome thanks', 'Great thanks'
        ],
        'affirmation': [
            'Yes', 'Yeah', 'Yep', 'Sure', 'Of course', 'Please do', 'Yes please', 'Ok', 'Okay'
        ],
        'negation': [
            'No', 'Nope', 'Nah', 'No thanks', 'Not right now', 'Nevermind'
        ],
        'question_intro': [
            'I have a question', 'Can I ask a question?', 'I need to ask something', 'Got a question',
            'May I ask a question?', 'I got a question', 'Question for you', 'Can you answer a question?',
            'I have an inquiry', 'I want to ask something', 'Let me ask you a question'
        ]
    }

    queries = []
    intents_list = []
    for intent_name, query_list in synthetic_data.items():
        for query in query_list:
            queries.append(query)
            intents_list.append(intent_name)
    df = pd.DataFrame({'query': queries, 'intent': intents_list})

    # 2. Train Intent Classification Model
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['query'])
    y = df['intent']
    intent_classifier = LogisticRegression(max_iter=1000, random_state=42) # Added random_state for reproducibility
    intent_classifier.fit(X, y)

    return tfidf_vectorizer, intent_classifier

tfidf_vectorizer, intent_classifier = load_chatbot_components()

# 3. Sentiment Analysis Function using TextBlob
def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using TextBlob.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: 'Positive', 'Negative', or 'Neutral' based on polarity.
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Define a detailed knowledge base with factual answers and interactive follow-ups
KNOWLEDGE_BASE = [
    {
        "keywords": ["established", "founded", "history", "started"],
        "context_key": "established",
        "answer": "Women's University in Africa (WUA) was established in 2002 by two visionaries: Professor Hope Cynthia Sadza and Dr. Fay Chung.",
        "follow_up": "Would you like to know more about the founders or our mission?"
    },
    {
        "keywords": ["location", "located", "where", "address", "campus"],
        "context_key": "location",
        "answer": "Our main campus is located in Manresa, Harare, Zimbabwe. We also have other campuses in Bulawayo, Mutare, and Kadoma.",
        "follow_up": "Are you looking for directions to a specific campus?"
    },
    {
        "keywords": ["programme", "program", "degree", "offer", "course"],
        "context_key": "programme",
        "answer": "We offer a variety of undergraduate and postgraduate degree programmes across our faculties.",
        "follow_up": "Are you interested in a specific faculty or level of study?"
    },
    {
        "keywords": ["entry", "requirement", "qualify", "qualification"],
        "context_key": "entry",
        "answer": "General entry requirements for undergraduate programmes include 5 O-Level passes (including English Language) and at least 2 A-Level passes. Postgraduate requirements vary by programme.",
        "follow_up": "Would you like to speak to an admissions officer about your qualifications?"
    },
    {
        "keywords": ["male", "man", "men", "boy", "gender"],
        "context_key": "male",
        "answer": "Yes! WUA is open to male students. Our enrollment policy maintains a ratio of 80% women and 20% men to empower women while maintaining an inclusive environment.",
        "follow_up": "Do you have any other questions about our student demographic?"
    },
    {
        "keywords": ["residence", "res", "hostel", "accommodation"],
        "context_key": "accommodation",
        "answer": "Yes, we offer on-campus accommodation (hostels) as well as partnered off-campus boarding houses.",
        "follow_up": "Would you like to know how to apply for a space in the residence?"
    },
    {
        "keywords": ["faculty", "faculties"],
        "context_key": "faculty",
        "answer": "WUA has several faculties, including: Agricultural Sciences, Management & Entrepreneurial Development, Social & Gender Transformative Sciences, and Information Technology.",
        "follow_up": "Which faculty's programmes are you most interested in?"
    },
    {
        "keywords": ["src", "student representative council", "election"],
        "context_key": "src",
        "answer": "The Student Representative Council (SRC) is the voice of the students. SRC applications and elections are held annually.",
        "follow_up": "Are you interested in running for an SRC position or voting?"
    },
    {
        "keywords": ["fee", "account", "payment", "tuition", "pay"],
        "context_key": "fee",
        "answer": "The tuition fee for the current semester is $775 for undergraduate programs and $1,000 for postgraduate programs. Other associated fees (like lab or library fees) usually total around $30. For more info on fees and payment plans, visit https://www.wua.ac.zw/fees-and-finance/", 
        "follow_up": "Would you like to know about our flexible payment plans or available scholarships?"
    },
    {
        "keywords": ["deadline", "due date"],
        "context_key": "deadline",
        "answer": "Fall semester registration: August 1st. Spring semester registration: January 15th. Summer registration: May 1st. Late registration incurs a $50 fee and must be approved by your department head.", 
        "follow_up": "Do you need help navigating the [registration portal](https://myhope.wua.ac.zw/)?"
    },
    {
        "keywords": ["scholarship", "grant", "bursary"],
        "context_key": "scholarship",
        "answer": "We offer several merit-based and need-based scholarships. The typical deadline to apply for financial aid is March 1st.", 
        "follow_up": "Would you like me to guide you on how to submit a scholarship application?"
    },
    {
        "keywords": ["financial aid", "loan", "work-study"],
        "context_key": "financial aid",
        "answer": "Financial aid options include university grants, federal loans (FAFSA required), and work-study programs. You can contact the financial aid office at finaid@wua.edu.", 
        "follow_up": "Are you interested in work-study opportunities on campus?"
    },
    {
        "keywords": ["apply", "application", "admission"],
        "context_key": "apply",
        "answer": "To apply, submit your form online via our portal at https://www.wua.ac.zw/how-to-apply/. Required documents include official transcripts, two letters of recommendation, personal essays, and proof of identity. Application fee is $25. Deadlines vary: Undergraduate - March 1st, Graduate - June 1st.", 
        "follow_up": "Are you an international student or a domestic applicant?"
    },
    {
        "keywords": ["library", "book"],
        "context_key": "library",
        "answer": "The main campus library is open Monday to Friday from 8:00 AM to 10:00 PM, and weekends from 10:00 AM to 6:00 PM. You can borrow up to 5 books at a time.", 
        "follow_up": "Do you need instructions on how to access our online research databases?"
    },
    {
        "keywords": ["campus", "tour", "facility"],
        "context_key": "campus",
        "answer": "Our main campus features modern science labs, three comprehensive libraries, and extensive sports facilities including an Olympic-sized swimming pool.", 
        "follow_up": "Would you like to register for a guided campus tour?"
    },
    {
        "keywords": ["registration", "register", "enroll", "enrollment"],
        "context_key": "registration",
        "answer": "Fall semester registration: August 1st. Spring semester registration: January 15th. Summer registration: May 1st. Late registration incurs a $100 fee and must be approved by your department head.", 
        "follow_up": "Are you having trouble logging into the Student Portal? [Student Portal](https://myhope.wua.ac.zw/)"
    },
    {
        "keywords": ["timetable", "schedule", "class time"],
        "context_key": "timetable",
        "answer": "Your personalized class timetable is available in the Student Information System on the [portal](https://myhope.wua.ac.zw/). It usually updates one week before the semester starts.", 
        "follow_up": "Are you currently experiencing a timezone challenge with your classes?"
    },
    {
        "keywords": ["exam", "test", "final", "midterm"],
        "context_key": "exam",
        "answer": "Final exam schedules are published midway through the semester on the university notice board and the [student portal](https://myhope.wua.ac.zw/).", 
        "follow_up": "Would you like some resources or tips on exam preparation?"
    },
    {
        "keywords": ["support", "help", "counseling", "counselling"],
        "context_key": "support",
        "answer": "Academic tutoring, writing center, disability services, mental health counseling, and student advocacy. Most services free for enrolled students. Visit studentservices@wua.edu or call 555-0789.", 
        "follow_up": "Is there a specific type of support you feel would be most helpful right now? [Student Support](https://www.wua.ac.zw/contact-us/)"
    },
    {
        "keywords": ["portal", "login", "account access"],
        "context_key": "portal",
        "answer": "You can access grades, registration, and forms on the [WUA Student Portal](https://myhope.wua.ac.zw/).", 
        "follow_up": "Do you need help logging in?"
    },
    {
        "keywords": ["website"],
        "context_key": "website",
        "answer": "You can find more general information about our programs and faculty on the [official WUA website](http://www.wua.ac.zw).", 
        "follow_up": "Is there a specific department or program you are searching for?"
    },
    {
        "keywords": ["career", "job", "internship", "resume"],
        "context_key": "career",
        "answer": "Resume workshops monthly. Job fairs held twice yearly (Fall and Spring). Internship placements with 200+ employers. Interview preparation and career counseling. Contact careers@wua.edu or visit Career Services in Building B.", 
        "follow_up": "Would you like to schedule an appointment with a career counselor?"
    },
    {
        "keywords": ["health", "clinic", "medical"],
        "context_key": "health",
        "answer": "Free clinic on campus open Monday-Friday 8 AM-5 PM, Saturday 10 AM-2 PM. Medical consultations, vaccinations, and mental health counseling available. Book appointments via health portal or call 555-0123.", 
        "follow_up": "Do you need the link to the health portal? [Health Portal](https://www.wua.ac.zw/physical-support/)"
    },
    {
        "keywords": ["international", "visa", "foreign student"],
        "context_key": "international",
        "answer": "Visa support and I-20 processing assistance. Orientation program in August. Cultural events and international student club. English language support available. Contact intl@wua.edu. Or read explore more on https://www.wua.ac.zw/international-study/", 
        "follow_up": "Are you an incoming international student? We'd be delighted to have you at WUA"
    },
    {
        "keywords": ["transfer", "transfer credit"],
        "context_key": "transfer",
        "answer": "Up to 60 transfer credits accepted from accredited institutions. Submit official transcripts and course descriptions. Minimum GPA 2.5 for transfer eligibility. Apply by May 1st for Fall admission.", 
        "follow_up": "Would you like to speak to a transfer advisor? [Transfer Advisor](https://www.wua.ac.zw/transfer-students/)"
    },
    {
        "keywords": ["calendar", "dates"],
        "context_key": "calendar",
        "answer": "See the detailed calendar on the school website on https://www.wua.ac.zw/wp-content/uploads/2025/12/2026-RETURNING-STUDENTS-CALENDAR.pdf", 
        "follow_up": "Are you looking for a specific date?"
    },
    {
        "keywords": ["parking", "permit"],
        "context_key": "parking",
        "answer": "Parking permits required for all vehicles. Costs: Student annual permit $200, semester permit $120. Buy online at portal.wua.edu. Violations: $25 fine. Lot locations: A-North, B-South, C-West.", 
        "follow_up": "Do you need a link to purchase a permit?"
    },
    {
        "keywords": ["technology", "it desk", "wifi", "internet", "tech support"],
        "context_key": "technology",
        "answer": "IT Help Desk: Building C, Room 101. Email: tech@wua.edu. Phone: 555-0456. Operating hours: Monday-Friday 8 AM-6 PM, Saturday 10 AM-4 PM. Campus-wide free Wi-Fi available everywhere.", 
        "follow_up": "Are you having trouble connecting to the campus Wi-Fi?"
    },
    {
        "keywords": ["grading", "grade", "gpa", "marks"],
        "context_key": "grading",
        "answer": "A (90-100), B (80-89), C (70-79), D (60-69), F (Below 60). Minimum passing grade for major courses: C. GPA calculated on 4.0 scale. Grade appeals must be submitted within 2 weeks of grade posting.", 
        "follow_up": "Do you need help calculating your current GPA?"
    },
    {
        "keywords": ["attendance", "absent", "missed class"],
        "context_key": "attendance",
        "answer": "Attend all classes. Missing more than 3 sessions may result in course withdrawal. Excused absences require documentation. Contact your professor immediately if unable to attend.", 
        "follow_up": "Have you missed any consecutive classes recently?"
    },
    {
        "keywords": ["academic registry", "registry"],
        "context_key": "academic registry",
        "answer": "The Academic Registry is located in Building A, Room 102. They are open from 8 AM to 4 PM, Monday to Friday.", 
        "follow_up": "Would you like me to find their contact email for you?"
    },
    {
        "keywords": ["defer", "deferring", "deferment", "drop", "dropout", "leave"],
        "context_key": "defer",
        "answer": "Deferring or dropping your studies is a significant decision. We are here to support you through the process or help you find alternatives.",
        "follow_up": "To ensure an advisor can follow up and provide you with your options, please reply with your Student ID (e.g., W222023)."
    }
]


# 4. Chatbot Response Logic
def generate_chatbot_response(query, tfidf_vec, intent_clf, context=None):
    """
    Generates a chatbot response based on predicted intent, sentiment, and conversation context.

    Args:
        query (str): The user's input query.
        tfidf_vec (TfidfVectorizer): Trained TF-IDF vectorizer.
        intent_clf (LogisticRegression): Trained intent classification model.
        context (str): The context keyword from the previous turn (if waiting for a yes/no).

    Returns:
        tuple: (str, str) The tailored chatbot response, and the new context.
    """
    query_lower = query.lower()

    # Check for direct student ID input when context is defer
    if context == 'defer':
        student_id = query.strip().upper()
        
        # If user cancels by saying no
        import re
        if re.fullmatch(r"no|nope|nah|nevermind|cancel", query_lower.strip().strip(".!?,;'\"")):
            return "No problem! Let me know if you change your mind or need any other help.", None
            
        if len(student_id) >= 5:
            import csv
            from datetime import datetime
            import uuid
            import os
            
            alert_path = os.path.join(os.path.dirname(__file__), 'risk_alert_summary.csv')
            file_exists = os.path.isfile(alert_path)
            
            # Generate a unique Alert ID
            alert_id = f"A{(len(student_id) * 313) % 9999:04d}" # Simple pseudo-random string for alert ID
            
            try:
                with open(alert_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Alert ID', 'Student ID', 'Issue', 'Date Reported', 'Status'])
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    writer.writerow([alert_id, student_id, 'Requested Deferment/Dropout Info', current_date, 'Open'])
                    
                return f"Thank you. A flag has been raised for Student ID {student_id}. An advisor will review your profile and contact you regarding your options. Is there anything else I can help you with?", None
            except Exception as e:
                return "Thank you. However, there was an error processing your request. Please contact the administration directly. Is there anything else I can help you with?", None
        else:
            return "That doesn't look like a valid WUA Student ID. Please provide your Student ID, or say 'No' to cancel.", "defer"

    knowledge_base = KNOWLEDGE_BASE


    # Fallback to the AI Intent and Sentiment model for conversational/generic queries
    query_vectorized = tfidf_vec.transform([query])
    predicted_intent = intent_clf.predict(query_vectorized)[0]

    # Override for strict negation to prevent 'no thanks' from being misclassified as 'thanks'
    import re
    if re.fullmatch(r"no|nope|nah|no thanks|no thank you|nothing|none|that's all|that is all|no more", query_lower.strip().strip(".!?,;'\"")):
        predicted_intent = 'negation'
    sentiment = get_sentiment(query)
    
    # Handle conversation context answers (Yes/No to follow-ups)
    if context:
        new_context = None # Clear context after handling
        if predicted_intent == 'affirmation':
            if context == 'fee' or context == 'financial aid' or context == 'scholarship':
                return "Great! For flexible payment plans, please email studentaccounts@wua.edu. For scholarships, visit the [Financial Aid page](https://www.wua.ac.zw/fees-and-finance/) to download the application form.", new_context
            elif context == 'deadline' or context == 'registration' or context == 'portal':
                return "The [Student Portal](https://myhope.wua.ac.zw/) requires your Student ID and chosen password. If you forgot your password, click 'Forgot Password' on the login screen.", new_context
            elif context == 'apply' or context == 'application' or context == 'international':
                return "Awesome! You can start your application right now at [Apply to WUA](https://www.wua.ac.zw/how-to-apply/). If you are an international student, please also email intl@wua.edu for visa guidance.", new_context
            elif context == 'library':
                return "To access online databases, go to the [Library Portal](http://library.wua.ac.zw) and log in with your student credentials. EBSCOhost and JSTOR are our primary databases.", new_context
            elif context == 'campus':
                return "To sign up for a guided tour, please email admissions@wua.edu and specify if you'd prefer a weekday or weekend slot.", new_context
            elif context == 'timetable' or context == 'attendance':
                return "If you have a clash or missed consecutive classes, please reach out to the Academic Registry immediately to prevent any penalties.", new_context
            elif context == 'academic registry':
                return "Their email is registry@wua.edu and their phone number is 555-0800.", new_context
            elif context == 'exam':
                return "You can find past exam papers in the library repository, and the Academic Writing Center runs test-prep sessions every Friday at 2 PM.", new_context
            elif context == 'support':
                return "For any support needs, feel free to drop by the Student Services Center in Building A, or call the 24/7 hotline at 555-0789.", new_context
            elif context == 'career':
                return "To schedule an appointment, log into Handshake using your student email or call Career Services at 555-0199.", new_context
            elif context == 'health':
                return "Here is the link to the [Health Portal](https://www.wua.ac.zw/physical-support/). You can book appointments directly online.", new_context
            elif context == 'transfer':
                return "You can reach a transfer advisor directly at transfers@wua.edu or by calling 555-0211.", new_context
            elif context == 'technology':
                return "If you are having Wi-Fi issues, try forgetting the 'WUA-Students' network and reconnecting using your student portal login. If that fails, visit the IT desk in Building C.", new_context
            elif context == 'grading':
                return "To calculate your GPA, divide the total number of grade points earned by the total number of credit hours attempted.", new_context
            elif context == 'established':
                return "The University was established to address gender disparity in higher education. For more history, you can visit the About Us section on our website.", new_context
            elif context == 'location':
                return "You can find maps and directions to all our campuses on the interactive map section of our [official WUA website](http://www.wua.ac.zw).", new_context
            elif context == 'programme' or context == 'faculty':
                return "You can view the full list of programmes and faculties in our prospectus at https://www.wua.ac.zw/prospectus/.", new_context
            elif context == 'entry':
                return "Please email admissions@wua.edu with your current qualifications and they will gladly assist you.", new_context
            elif context == 'male':
                return "We believe empowering women involves everyone! Male students are a valued part of our community.", new_context
            elif context == 'accommodation':
                return "You can apply for residence via the student portal. Spots are limited and assigned on a first-come, first-served basis.", new_context
            elif context == 'src':
                return "Look out for announcements regarding SRC nominations and elections on the student portal noticeboard at the start of the academic year.", new_context
            else:
                return "Okay, I'm glad I could help! Is there anything else you need?", new_context
                
        elif predicted_intent == 'negation':
            return "No problem! Have a great day!", new_context
            
        else:
            # If they didn't say yes/no but changed the subject
            pass # Continue to regular intent matching

    # Check for direct, factual matches in the knowledge base first
    import re
    # Track the best context_key and info match based on keyword sequence
    for entry in knowledge_base:
        for keyword in entry["keywords"]:
            # Match keyword or its simple plural (keyword + 's' optionally)
            if re.search(rf"\b{keyword}s?\b", query_lower):
                # Remember the context_key for the next turn
                return f"{entry['answer']} {entry['follow_up']}", entry["context_key"]

    responses = {
        'deadlines': {
            'Positive': "That's great you're on top of your deadlines! For specific assignment deadlines, please check your course syllabus or the university portal. Let me know if you need help navigating them!",
            'Neutral': "Deadlines can be tricky. Could you specify which deadline you're asking about? (e.g., assignments, registration, fee payment). I can then provide more precise information.",
            'Negative': "I understand that deadlines can be stressful. To help you best, could you tell me which deadline you're concerned about? We can look for solutions together."
        },
        'registration': {
            'Positive': "Fantastic! It's great you're preparing for registration. You can usually register through the student portal. What specifically would you like to know about the process?",
            'Neutral': "For course registration, please visit the academic portal. If you're having trouble or have specific questions about course selection, let me know!",
            'Negative': "I'm sorry to hear you're having issues with registration. Please describe the problem you're encountering, and I'll do my best to guide you to the right support or resources."
        },
        'library': {
            'Positive': "Wonderful! Our library has excellent resources. Are you looking for opening hours, how to borrow books, or perhaps research assistance?",
            'Neutral': "The university library offers a wide range of resources. Do you need information on opening hours, finding books, or accessing online databases?",
            'Negative': "I understand you might be having difficulties with the library. Please tell me what's wrong, and I'll help you find a solution or direct you to library staff."
        },
        'fees': {
            'Positive': "Great planning ahead! Information on tuition fees and payment methods can be found on the finance department's website. Is there a specific aspect you'd like to clarify?",
            'Neutral': "For information on university fees and payment schedules, please refer to the finance section of the official university website. What specifically are you looking for?",
            'Negative': "I understand that fee-related concerns can be frustrating. Please specify your issue with fees, and I'll connect you with the appropriate financial aid or accounting services."
        },
        'exams': {
            'Positive': "Excellent! Preparing for exams is key. Are you looking for the exam schedule, past papers, or study tips?",
            'Neutral': "Exam schedules and regulations are typically posted on the academic affairs page. Do you need help finding your specific exam timetable or other exam-related information?",
            'Negative': "It sounds like you're having trouble with exams. Please tell me more about your concern, whether it's about scheduling conflicts, retakes, or study anxiety, so I can assist you."
        },
        'support': {
            'Positive': "I'm glad you're reaching out for support! We have various services available. Are you seeking academic, mental health, career, or technical assistance?",
            'Neutral': "Our student support services cover a wide range of needs. Please specify the type of support you're looking for (e.g., academic, counseling, career) so I can guide you.",
            'Negative': "I'm really sorry to hear you're struggling. It's brave of you to ask for help. Please tell me more about what kind of support you need, and I'll connect you with the right department immediately."
        },
        'timetable': {
            'Positive': "Wonderful! Keeping track of your timetable is important. You can usually find your personalized schedule on the student portal. Need help accessing it?",
            'Neutral': "For your class timetable, please log into the student information system or check the departmental schedule. What specifically about the timetable are you trying to find?",
            'Negative': "Oh no, I understand timetable issues can be really disruptive. Please describe what's wrong with your timetable (e.g., clash, missing classes), and I'll help you find a solution."
        },
        'greeting': {
            'Positive': "Hello there! How can I assist you today?",
            'Neutral': "Hi! What can I help you with?",
            'Negative': "Hello. How can I help you out today?"
        },
        'thanks': {
            'Positive': "You're very welcome! Have a good day!",
            'Neutral': "You're welcome! Let me know if you need anything else.",
            'Negative': "You're welcome. Have a good day!"
        },
        'affirmation': {
            'Positive': "Great! What else can I do for you?",
            'Neutral': "Okay, what's next?",
            'Negative': "Understood. How else can I help?"
        },
        'negation': {
            'Positive': "No worries! Have a fantastic day!",
            'Neutral': "Alright. Let me know if you need anything else later. Have a good day!",
            'Negative': "Okay. Just let me know if you change your mind. We are here to support your academic journey."
        },
        'question_intro': {
            'Positive': "Sure, I'd love to help! What's your question?",
            'Neutral': "Sure, how may I help you?",
            'Negative': "Of course, I'm here to help. What is your question?"
        }
    }

    # Retrieve response based on predicted intent and sentiment
    generic_response = responses.get(predicted_intent, {}).get(sentiment, responses.get(predicted_intent, {}).get('Neutral', "I'm not sure how to respond to that. Could you rephrase your query?"))
    
    # Simple conversational items don't need hooks
    if predicted_intent in ['greeting', 'affirmation', 'question_intro']:
        return generic_response, None
        
    interactive_hook = " Is there anything else I can help you with?"
        
    if predicted_intent in ['thanks', 'negation']:
        if context in ['timetable', 'registration']:
            if predicted_intent == 'thanks':
                return f"You're very welcome! Best wishes on your studies!{interactive_hook}", None
            else:
                return "No worries! Have a fantastic day and best wishes on your studies!", None
        else:
            if predicted_intent == 'thanks':
                return f"You're very welcome! Have a good day!{interactive_hook}", None
            else:
                return "No worries! Have a good day!", None
        
    # Append a conversational hook for generic knowledge questions
    interactive_hook = " Is there anything else I can help you with?"
    return f"{generic_response}{interactive_hook}", None


# --- Dropout Predictor Components (Cached to run once) ---
@st.cache_resource
def load_dropout_model():
    """
    Loads the trained Logistic Regression model for dropout prediction from 'dropout_model.joblib'.
    This function is cached to run only once.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'dropout_model.joblib')
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading dropout model: {e}. Please ensure the model file is not corrupted.")
            return None
    else:
        st.warning(f"Dropout model file '{model_path}' not found. Please ensure the model is trained and saved in the current directory.")
        return None

dropout_model = load_dropout_model()


# --- Insights Components (Cached to run once for data generation) ---
@st.cache_data
def load_dropout_data_for_insights(n_students=1000):
    """
    Generates synthetic student data for dropout prediction insights.
    This function is cached to run only once.

    Args:
        n_students (int): Number of synthetic student records to generate.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic student data with dropout risk.
    """
    rng = np.random.default_rng(42)
    gpa = rng.uniform(1.0, 4.0, n_students)
    attendance = rng.integers(50, 101, n_students)
    missed_assignments = rng.integers(0, 11, n_students)
    study_hours = rng.uniform(1.0, 40.0, n_students)
    financial_aid = rng.integers(0, 2, n_students)  # 0 = no aid, 1 = aid
    engagement_score = rng.uniform(0.0, 10.0, n_students)  # 0 = disengaged, 10 = highly engaged
    clubs_sports = rng.integers(0, 2, n_students)           # 0 = not involved, 1 = involved

    dropout_risk = []
    for i in range(n_students):
        risk_score = 0
        if gpa[i] < 2.5:               risk_score += 1
        if attendance[i] < 70:         risk_score += 1
        if missed_assignments[i] > 5:  risk_score += 1
        if study_hours[i] < 10:        risk_score += 1
        if financial_aid[i] == 0:      risk_score += 0.5
        if engagement_score[i] < 4.0:  risk_score += 0.75
        if clubs_sports[i] == 0:       risk_score += 0.25
        dropout_risk.append(1 if risk_score >= 2.5 else 0)

    data = {
        'GPA': gpa,
        'Attendance_Percentage': attendance,
        'Missed_Assignments': missed_assignments,
        'Study_Hours_Per_Week': study_hours,
        'Financial_Aid': financial_aid,
        'Engagement_Score': engagement_score,
        'Clubs_Sports': clubs_sports,
        'Dropout_Risk': dropout_risk
    }
    return pd.DataFrame(data)

df_dropout_insights = load_dropout_data_for_insights()


# --- Authentication and Views ---

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None



if not st.session_state.logged_in:
    st.header("Welcome to Edupulse")
    st.markdown("Please log in to continue.")

    with st.form('login_form'):
        role_choice = st.selectbox('I am a/an', ['Student', 'Admin'])
        password = ''
        if role_choice == 'Admin':
            password = st.text_input('Admin password', type='password')
        submit = st.form_submit_button('Login')

    if submit:
        if role_choice == 'Admin':
            if password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.role = 'Admin'
                st.success('Logged in as Admin')
                safe_rerun()
            else:
                st.error('Invalid admin password')
        else:
            st.session_state.logged_in = True
            st.session_state.role = 'Student'
            st.success('Logged in as Student')
            safe_rerun()

else:
    # Top-level navigation and logout
    st.sidebar.write(f"Logged in as: {st.session_state.role}")
    if st.sidebar.button('Logout'):
        logout()

    # Home / Dashboard
    st.header('Home — Edupulse Dashboard')

    if st.session_state.role == 'Student':
        st.sidebar.subheader("Student Menu")
        student_menu = st.sidebar.radio("Navigation", [
            "Ask a question", 
            "Chat history", 
            "Browse FAQs", 
            "Quick links"
        ])

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [{"role": "assistant", "content": "Hi, I'm EduBot. How can I help you?"}]
        if 'chat_context' not in st.session_state:
            st.session_state.chat_context = None

        if student_menu == "Ask a question":
            st.subheader('Student Support Chatbot')
            
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message['role']):
                    st.markdown(message['content'])
                    # Provide feedback for assistant responses only after a "thank you"
                    if message['role'] == 'assistant' and message.get('show_feedback', False):
                        st.markdown("<p style='font-size: 0.85em; margin-bottom: 0px;'>Was this helpful?</p>", unsafe_allow_html=True)
                        f_col1, f_col2, f_col3 = st.columns([1, 1, 8])
                        fb_key = f"fb_{i}"
                        if st.session_state.get(fb_key) is None:
                            with f_col1:
                                if st.button("👍", key=f"up_{i}"):
                                    st.session_state[fb_key] = "yes"
                                    safe_rerun()
                            with f_col2:
                                if st.button("👎", key=f"down_{i}"):
                                    st.session_state[fb_key] = "no"
                                    safe_rerun()
                        else:
                            st.caption("Thanks for your feedback!")
                    
            prompt_input = st.chat_input('Ask anything about WUA...')
            
            prompt = None
            if prompt_input:
                prompt = prompt_input
            elif 'quick_query' in st.session_state and st.session_state.quick_query:
                prompt = st.session_state.quick_query
                st.session_state.quick_query = None

            if prompt:
                st.session_state.chat_history.append({'role':'user','content':prompt})
                with st.chat_message('user'):
                    st.markdown(prompt)
                with st.spinner('Thinking...'):
                    response, new_context = generate_chatbot_response(prompt, tfidf_vectorizer, intent_classifier, st.session_state.chat_context)
                    st.session_state.chat_context = new_context
                    with st.chat_message('assistant'):
                        st.markdown(response)
                
                # Check if user message is thanking the bot
                is_thanks = any(word in prompt.lower() for word in ['thank', 'thx', 'appreciate', 'helpful'])
                
                st.session_state.chat_history.append({'role':'assistant','content':response, 'show_feedback': is_thanks})
                safe_rerun()

        elif student_menu == "Chat history":
            st.subheader("Your Chat History")
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])
            if len(st.session_state.chat_history) <= 1:
                st.info("No chat history yet. Go to 'Ask a question' to start chatting!")

        elif student_menu == "Browse FAQs":
            st.subheader("Frequently Asked Questions")
            half_len = len(KNOWLEDGE_BASE) // 2
            for entry in KNOWLEDGE_BASE[:half_len]:
                keyword_str = ", ".join(entry['keywords'])
                with st.expander(f"Q: About {entry['context_key'].title()}"):
                    st.markdown(f"**Keywords:** {keyword_str}")
                    st.markdown(f"**Answer:** {entry['answer']}")
                    st.markdown(f"**Follow-up:** {entry['follow_up']}")

        elif student_menu == "Quick links":
            st.subheader("Quick Links")
            st.markdown("Here are some useful links to access university resources directly:")
            
            st.markdown("""
            * **[WUA Official Website](https://www.wua.ac.zw/)** - General information about programs and faculties
            * **[Student Portal](https://myhope.wua.ac.zw/)** - Access grades, registration, and forms
            * **[Admissions & Application](https://www.wua.ac.zw/how-to-apply/)** - Start or manage your application
            * **[Library Portal](http://library.wua.ac.zw)** - Access online databases and research materials
            * **[Health & Physical Support](https://www.wua.ac.zw/physical-support/)** - Access health resources
            * **[Prospectus & Programmes](https://www.wua.ac.zw/prospectus/)** - View available courses and faculties
            """)


    else:
        st.sidebar.subheader("Admin Menu")
        admin_menu = st.sidebar.radio("Navigation", [
            "Overview", 
            "Student list", 
            "Attendance records", 
            "Risk report", 
            "Risk alert summary"
        ])

        if admin_menu == "Overview":
            # Admin: full dashboard with tabs (no student chatbot)
            dropout_tab, insights_tab = st.tabs(['Dropout Predictor', 'Insights'])

            # Dropout Predictor tab
            with dropout_tab:
                st.header('Student Dropout Risk Predictor')
                st.markdown('Use this tool to predict the likelihood of a student dropping out based on their academic and engagement metrics.')

                # --- CSV STUDENT DATABASE LOOKUP ---
                st.write("### Student Details Lookup")

                # --- SINGLE STUDENT ID INPUT ---
                # Accepts 6 digits (222023) typed manually OR the full ID pasted from a table (W222023)
                raw_id_input = st.text_input(
                    "Enter or paste Student ID",
                    max_chars=7,
                    placeholder="e.g. 222023 or W222023",
                    key="student_id_field",
                    help="You can type the 6-digit number or paste the full ID (e.g. W222023) directly from a table."
                )

                student_id_input = ""
                if raw_id_input:
                    # Strip the W prefix if the user pasted the full ID (e.g. W222023)
                    digits_part = raw_id_input.strip()
                    if digits_part.upper().startswith("W"):
                        digits_part = digits_part[1:]

                    if not digits_part.isdigit():
                        st.error("❌ Invalid input: The numeric part of the ID must contain digits only (0–9). No letters or symbols allowed.")
                    elif len(digits_part) < 6:
                        st.error(f"❌ Too short: Please enter all 6 digits. You've entered {len(digits_part)} digit(s).")
                    elif len(digits_part) > 6:
                        st.error(f"❌ Too long: Student ID must be exactly 6 digits after the W prefix. Got {len(digits_part)}.")
                    else:
                        student_id_input = f"W{digits_part}".upper()

                if student_id_input:
                    csv_path = os.path.join(os.path.dirname(__file__), 'students.csv')
                    if os.path.exists(csv_path):
                        try:
                            # Read CSV as string to avoid stripping zeros if any exist in IDs
                            df_students = pd.read_csv(csv_path, dtype=str)
                            # Ensure we clean whitespace from column names just in case
                            df_students.columns = df_students.columns.str.strip()

                            # Find the row matching the Student ID exactly (case-insensitive search)
                            match = df_students[df_students['Student ID'].str.upper() == student_id_input]

                            if not match.empty:
                                student_info = match.iloc[0]
                                st.info(f"**Name:** {student_info.get('Student Name', '')} {student_info.get('Student  Surname', student_info.get('Student Surname', ''))} | **Programme:** {student_info.get('Student Programme', '')} | **Year:** {student_info.get('Academic Year', '')}")
                            else:
                                st.warning("⚠️ Student ID not found in our database. You can still manually enter metrics below.")
                        except Exception as e:
                            st.error(f"Error reading students database: {e}")
                    else:
                        st.warning("The 'students.csv' file could not be found in the application directory. Using manual entry only.")

                st.divider()

                if dropout_model is not None:
                    st.write('### Enter student metrics to predict dropout risk:')
                    col1, col2 = st.columns(2)
                    with col1:
                        gpa = st.slider('GPA (Grade Point Average)', min_value=1.0, max_value=4.0, value=2.5, step=0.1)
                        missed_assignments = st.slider('Missed Assignments (total)', min_value=0, max_value=20, value=5, step=1)
                        engagement_score = st.slider('Student Engagement Score (0 = disengaged, 10 = highly engaged)', min_value=0.0, max_value=10.0, value=5.0, step=0.5)
                        financial_aid = st.selectbox('Receiving Financial Aid?', options=[0,1], format_func=lambda x: 'Yes' if x==1 else 'No')
                    with col2:
                        attendance = st.slider('Attendance Percentage (%)', min_value=0, max_value=100, value=75, step=1)
                        study_hours = st.slider('Study Hours Per Week', min_value=0.0, max_value=60.0, value=15.0, step=0.5)
                        clubs_sports = st.selectbox('Involved in Clubs / Sports?', options=[0,1], format_func=lambda x: 'Yes' if x==1 else 'No')

                    if st.button('Predict Dropout Risk'):
                        input_data = pd.DataFrame([{
                            'GPA': gpa,
                            'Attendance_Percentage': attendance,
                            'Missed_Assignments': missed_assignments,
                            'Study_Hours_Per_Week': study_hours,
                            'Financial_Aid': financial_aid,
                            'Engagement_Score': engagement_score,
                            'Clubs_Sports': clubs_sports
                        }])
                        prediction = dropout_model.predict(input_data)
                        prediction_proba = dropout_model.predict_proba(input_data)[:,1]
                        st.subheader('Prediction Results:')
                        
                        risk_level = 'High' if prediction[0] == 1 else 'Low'
                        if risk_level == 'High':
                            st.error(f"**High Dropout Risk** (Probability: {prediction_proba[0]:.2f})")
                        else:
                            st.success(f"**Low Dropout Risk** (Probability: {prediction_proba[0]:.2f})")

                        # Append to risk report
                        if student_id_input:
                            report_path = os.path.join(os.path.dirname(__file__), 'class_risk_report.csv')
                            try:
                                import csv
                                from datetime import datetime
                                file_exists = os.path.isfile(report_path)
                                with open(report_path, mode='a', newline='') as f:
                                    writer = csv.writer(f)
                                    if not file_exists:
                                        writer.writerow(['Date', 'Student ID', 'GPA', 'Missed_Assignments', 'Financial_Aid', 'Study_Hours', 'Engagement_Score', 'Clubs_Sports', 'Risk_Score', 'Risk_Level'])

                                    current_date = datetime.now().strftime('%Y-%m-%d')
                                    writer.writerow([current_date, student_id_input, gpa, missed_assignments, financial_aid, study_hours, engagement_score, clubs_sports, round(prediction_proba[0], 2), risk_level])
                                st.info(f"Student {student_id_input}'s metrics recorded to Risk Report.")
                            except Exception as e:
                                st.error(f"Could not save to class_risk_report.csv: {e}")
                        else:
                            st.info("Results not saved. Enter a complete 6-digit Student ID string above to associate and save these metrics.")

                        # --- HOW TO HELP / ACTIONABLE SUGGESTIONS ---
                        st.subheader(" How to Help / Action Plan")
                        suggestions = []
                    
                        if study_hours < 10.0:
                            suggestions.append(
                                "**Low Study Hours:** The student is dedicating less than 10 hours weekly to coursework. "
            "Encourage them to create a structured timetable, breaking study sessions into manageable blocks "
            "(e.g., 2 hours daily). Suggest using campus resources like study groups or the library to build consistency. "
            "Highlight the importance of balancing rest with focused study to avoid burnout."
        )

                        if attendance < 75:
                            suggestions.append(
                                "**Poor Attendance:** Attendance below 75% may affect comprehension and grades. "
            "Reach out to discuss possible barriers — personal, health, or transportation issues. "
            "Offer flexible solutions such as online lecture recordings, carpooling options, or counseling support. "
            "Reinforce how regular attendance strengthens understanding and builds rapport with instructors."
        )

                        if missed_assignments >= 4:
                            suggestions.append(
                                "**Missed Assignments:** Missing 4 or more assignments signals workload management challenges. "
            "Recommend visiting the Academic Writing Center or peer tutoring sessions to catch up. "
            "Encourage breaking tasks into smaller milestones and using digital planners or reminders. "
            "Offer reassurance that with structured support, they can regain momentum."
        )

                        if gpa < 2.5:
                            suggestions.append(
                                "**Low GPA:** A GPA below 2.5 may place the student at risk of academic probation. "
            "Schedule a mandatory advising meeting to review course load, study strategies, and available support services. "
            "Discuss options like reducing credit hours, enrolling in skill‑building workshops, or joining peer study groups. "
            "Frame this as an opportunity to reset and strengthen their academic path."
        )

                        if financial_aid == 0:
                            suggestions.append(
                                "**No Financial Aid:** The student currently has no financial aid support. "
                                "Encourage them to explore scholarships, grants, and work‑study opportunities. "
                                "Provide guidance on application deadlines and required documents. "
                                "Offer to connect them with the financial aid office for personalized assistance. "
                                "Stress that financial support can ease stress and allow them to focus more fully on studies."
                            )

                        if engagement_score < 4.0:
                            suggestions.append(
                                "**Low Engagement Score:** The student appears disengaged from the campus environment. "
                                "Schedule a one-on-one check-in to understand barriers (personal, social, or academic). "
                                "Introduce them to student mentorship programmes, study circles, or campus events. "
                                "Engaged students are significantly less likely to drop out — even small steps count."
                            )

                        if clubs_sports == 0:
                            suggestions.append(
                                "**Not Involved in Clubs or Sports:** Extra-curricular participation builds community "
                                "and a sense of belonging, which are key protective factors against dropout. "
                                "Recommend the student explores at least one club, society, or sports team. "
                                "The SRC noticeboard and student portal list all active societies."
                            )

                        if len(suggestions) > 0:
                            st.write("Based on the input metrics, here are targeted interventions for this student:")
                            for suggestion in suggestions:
                                st.write(f"- {suggestion}")
                        else:
                            st.info("The student's current academic and engagement metrics look stable. No immediate targeted interventions are required at this time.")

                else:
                    st.warning("The dropout prediction model could not be loaded. Please ensure 'dropout_model.joblib' is available.")

            # Insights tab
            with insights_tab:
                st.header('Student Dropout Insights')
                st.markdown('Visualize key features and their distribution relative to dropout risk to understand underlying patterns.')
                df_dropout_insights['Dropout_Risk_Category'] = df_dropout_insights['Dropout_Risk'].map({0: 'Low Risk', 1: 'High Risk'})

                st.subheader("Distribution of Dropout Risk")
                dropout_counts = df_dropout_insights['Dropout_Risk_Category'].value_counts().reset_index()
                dropout_counts.columns = ['Dropout Risk', 'Count']
                chart_dropout_counts = alt.Chart(dropout_counts).mark_bar().encode(
                    x=alt.X('Dropout Risk:N', sort='-y'),
                    y='Count:Q',
                    color=alt.Color('Dropout Risk:N', scale=alt.Scale(domain=['Low Risk', 'High Risk'], range=['#4CAF50', '#F44336']))
                ).properties(
                    title='Overall Distribution of Dropout Risk'
                ).interactive()
                st.altair_chart(chart_dropout_counts, width='stretch')

                # Chart 1: GPA distribution by Dropout Risk
                st.subheader("GPA Distribution by Dropout Risk")
                chart_gpa = alt.Chart(df_dropout_insights).mark_bar().encode(
                    alt.X('GPA:Q', bin=alt.Bin(maxbins=20), title='GPA'),
                    alt.Y('count()', title='Number of Students'),
                    color=alt.Color('Dropout_Risk_Category:N', title='Dropout Risk', scale=alt.Scale(domain=['Low Risk', 'High Risk'], range=['#4CAF50', '#F44336']))
                ).properties(
                    title='GPA Distribution by Dropout Risk'
                ).interactive()
                st.altair_chart(chart_gpa, width='stretch')

                # Chart 2: Attendance Percentage distribution by Dropout Risk
                st.subheader("Attendance Percentage Distribution by Dropout Risk")
                chart_attendance = alt.Chart(df_dropout_insights).mark_bar().encode(
                    alt.X('Attendance_Percentage:Q', bin=alt.Bin(maxbins=20), title='Attendance Percentage'),
                    alt.Y('count()', title='Number of Students'),
                    color=alt.Color('Dropout_Risk_Category:N', title='Dropout Risk', scale=alt.Scale(domain=['Low Risk', 'High Risk'], range=['#4CAF50', '#F44336']))
                ).properties(
                    title='Attendance Percentage Distribution by Dropout Risk'
                ).interactive()
                st.altair_chart(chart_attendance, width='stretch')

                # Chart 3: Missed Assignments vs. Dropout Risk
                st.subheader("Missed Assignments vs. Dropout Risk")
                chart_missed_assignments = alt.Chart(df_dropout_insights).mark_bar().encode(
                    alt.X('Missed_Assignments:Q', bin=alt.Bin(maxbins=10), title='Number of Missed Assignments'),
                    alt.Y('count()', title='Number of Students'),
                    color=alt.Color('Dropout_Risk_Category:N', title='Dropout Risk', scale=alt.Scale(domain=['Low Risk', 'High Risk'], range=['#4CAF50', '#F44336']))
                ).properties(
                    title='Missed Assignments Distribution by Dropout Risk'
                ).interactive()
                st.altair_chart(chart_missed_assignments, width='stretch')

                # Feature Importance from Logistic Regression Model (if available)
                if dropout_model is not None and hasattr(dropout_model, 'coef_'):
                    st.subheader("Feature Importance from Dropout Prediction Model")
                    coefficients = dropout_model.coef_[0]
                    # Prefer feature names stored in the model itself (set by sklearn when trained on a DataFrame)
                    if hasattr(dropout_model, 'feature_names_in_'):
                        feature_names = list(dropout_model.feature_names_in_)
                    else:
                        # Fall back to the full list, sliced to match the actual number of coefficients
                        all_feature_names = ['GPA', 'Attendance_Percentage', 'Missed_Assignments',
                                             'Study_Hours_Per_Week', 'Financial_Aid', 'Engagement_Score', 'Clubs_Sports']
                        feature_names = all_feature_names[:len(coefficients)]
                    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
                    # Sort by absolute coefficient value to show most influential features
                    feature_importance_df['Absolute_Coefficient'] = feature_importance_df['Coefficient'].abs()
                    feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

                    st.write("Coefficients from the Logistic Regression model, indicating the impact of each feature on dropout risk.")
                    st.dataframe(feature_importance_df[['Feature', 'Coefficient']])

                    chart_feature_importance = alt.Chart(feature_importance_df).mark_bar().encode(
                        x=alt.X('Coefficient:Q', title='Coefficient Value'),
                        y=alt.Y('Feature:N', sort='-x', title='Feature'),
                        color=alt.condition(
                            alt.datum.Coefficient > 0,
                            alt.value("steelblue"),
                            alt.value("orange")
                        )
                    ).properties(
                        title='Logistic Regression Feature Coefficients'
                    ).interactive()
                    st.altair_chart(chart_feature_importance, width='stretch')
                    st.markdown("""
                    **Interpretation of Coefficients:**
                    -   **Positive Coefficient**: Indicates that as the feature value increases, the likelihood of dropout increases.
                    -   **Negative Coefficient**: Indicates that as the feature value increases, the likelihood of dropout decreases.
                    -   The magnitude of the coefficient indicates the strength of the relationship.
                    """)
                elif dropout_model is not None:
                    st.info("Feature importance visualization is only available for Logistic Regression models with accessible coefficients.")
                else:
                    st.warning("Dropout model not loaded, cannot display feature importance.")
        elif admin_menu == "Student list":
            st.header("Student List")
            st.markdown("Overview of all enrolled students.")
            csv_path = os.path.join(os.path.dirname(__file__), 'students.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("students.csv not found.")

        elif admin_menu == "Attendance records":
            st.header("Attendance Records")
            st.markdown("Daily attendance tracking for enrolled students.")
            csv_path = os.path.join(os.path.dirname(__file__), 'attendance_records.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
                
                # Show quick chart
                st.subheader("Attendance Status Distribution")
                chart = alt.Chart(df).mark_arc().encode(
                    theta="count()",
                    color="Status:N"
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("attendance_records.csv not found.")

        elif admin_menu == "Risk report":
            st.header("Class Risk Report")
            st.markdown("Comprehensive view of risk metrics across the class.")
            csv_path = os.path.join(os.path.dirname(__file__), 'class_risk_report.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
                
                st.subheader("Risk Level Breakdown")
                chart = alt.Chart(df).mark_bar().encode(
                    x="Risk_Level:N",
                    y="count()",
                    color="Risk_Level:N"
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("class_risk_report.csv not found.")

        elif admin_menu == "Risk alert summary":
            st.header("Risk Alert Summary")
            st.markdown("Active alerts requiring immediate administrative attention.")
            csv_path = os.path.join(os.path.dirname(__file__), 'risk_alert_summary.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Highlight open alerts
                st.dataframe(df, use_container_width=True) # Streamlit dataframe handles simple rendering; styling via applymap often works but standard st.dataframe is safer
            else:
                st.warning("risk_alert_summary.csv not found.")
