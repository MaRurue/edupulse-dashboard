import sys
import os

file_path = os.path.join(r"c:\Users\CHARA\Downloads\Edupulse-2-0", "edupulse.py")
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

admin_tabs_start = content.find("        # Admin: full dashboard with tabs (no student chatbot)")

before_tabs = content[:admin_tabs_start]
tabs_content = content[admin_tabs_start:]

# Indent the whole tabs_content to put it inside the if block
indented_tabs = "\n".join(["    " + line if line.strip() else line for line in tabs_content.split("\n")])

new_admin_logic = """        if admin_menu == "My class overview":
""" + indented_tabs + """
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

        elif admin_menu == "Class risk report":
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
"""

new_content = before_tabs + new_admin_logic

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Admin view updated.")
