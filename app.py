import streamlit as st
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import lecture modules
from lectures import (
    lecture1_intro,
    lecture2_color_spaces,
    lecture3_pixel_ops,
    lecture4_filtering,
    lecture5_denoising,
    lecture6_edge_detection,
    lecture7_morphology,
    lecture8_geometric,
    lecture9_final_project
)

def main():
    st.set_page_config(
        page_title="Interactive Image Processing Lectures",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("محاضرات معالجة الصور التفاعلية")
    st.markdown("تعلم مفاهيم معالجة الصور من خلال عروض توضيحية تفاعلية عملية!")
    
    # Sidebar navigation
    st.sidebar.title("التنقل")
    
    lectures = {
        "المحاضرة الأولى: مقدمة": lecture1_intro,
        "المحاضرة الثانية: مساحات الألوان": lecture2_color_spaces,
        "المحاضرة الثالثة: عمليات البكسل": lecture3_pixel_ops,
        "المحاضرة الرابعة: الترشيح": lecture4_filtering,
        "المحاضرة الخامسة: إزالة الضوضاء": lecture5_denoising,
        "المحاضرة السادسة: كشف الحواف": lecture6_edge_detection,
        "المحاضرة السابعة: العمليات المورفولوجية": lecture7_morphology,
        "المحاضرة الثامنة: التحويلات الهندسية": lecture8_geometric,
        "المحاضرة التاسعة: المشروع النهائي": lecture9_final_project
    }
    
    selected_lecture = st.sidebar.selectbox(
        "اختر محاضرة:",
        list(lectures.keys())
    )
    
    # Display selected lecture
    lectures[selected_lecture].show()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("مبني باستخدام Streamlit و OpenCV و NumPy")

if __name__ == "__main__":
    main()
