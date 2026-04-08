import sys
import os

file_path = os.path.join(r"c:\Users\CHARA\Downloads\Edupulse-2-0", "edupulse.py")
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Extract knowledge_base to global KNOWLEDGE_BASE
kb_start = content.find("    # Define a detailed knowledge base")
kb_end = content.find("    query_lower = query.lower()")

kb_content = content[kb_start:kb_end]
kb_content_unindented = "\n".join([line[4:] if line.startswith("    ") else line for line in kb_content.split("\n")])
kb_content_unindented = kb_content_unindented.replace("knowledge_base =", "KNOWLEDGE_BASE =")

# Remove kb from original function and insert at module level
content = content[:kb_start] + "    knowledge_base = KNOWLEDGE_BASE\n\n" + content[kb_end:]

func_def_index = content.find("# 4. Chatbot Response Logic\ndef generate_chatbot_response")
content = content[:func_def_index] + kb_content_unindented + "\n" + content[func_def_index:]

# 2. Refactor Student Section
student_section_start = content.find("    if st.session_state.role == 'Student':\n        st.sidebar.subheader(\"Student Menu\")")
student_section_end = content.find("    else:\n        st.sidebar.subheader(\"Admin Menu\")")

student_section = """    if st.session_state.role == 'Student':
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
            
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.markdown(message['content'])
                    
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
                st.session_state.chat_history.append({'role':'assistant','content':response})

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
            st.markdown("Click any link below to ask EduBot about it instantly!")
            st.markdown(\"""
            <style>
            .stButton>button {
                padding: 2px 10px !important;
                font-size: 0.8rem !important;
                height: auto !important;
                min-height: 30px !important;
                width: auto !important;
                display: inline-flex !important;
            }
            div[data-testid="column"] .stButton {
                text-align: left;
            }
            </style>
            \""", unsafe_allow_html=True)
            col_btn, col_space = st.columns([1, 4])
            with col_btn:
                if st.button("Application", key="ql_app", use_container_width=False):
                    st.session_state.quick_query = "What is the application process?"
                if st.button("Register", key="ql_reg", use_container_width=False):
                    st.session_state.quick_query = "When is registration and how do I register?"
                if st.button("Portal", key="ql_portal", use_container_width=False):
                    st.session_state.quick_query = "How do I access the student portal?"
                if st.button("Website", key="ql_web", use_container_width=False):
                    st.session_state.quick_query = "What is the official school website?"
                if st.button("Exams", key="ql_exams", use_container_width=False):
                    st.session_state.quick_query = "Where can I find exam schedules?"
            
            st.info("Select 'Ask a question' from the side menu to see the bot's response!")

"""

content = content[:student_section_start] + student_section + "\n" + content[student_section_end:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
