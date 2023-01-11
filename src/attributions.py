import streamlit as st




def attribute():
    
    # for _ in range(20):
    #     st.sidebar.markdown('')
    
    st.markdown('''<center><span style="font-family:syne;font-weight:800; font-size:25px">Attributions</span></center>''',unsafe_allow_html=True)
    st.markdown("> <span style='font-family:Syne;'>Machinery Section</span>",unsafe_allow_html=True)
    content0 = '''The Images are selectively scraped from the Wikipedia using the WikiAPI.'''
    content1 = '''The Car Specific Data, such as the Engine, Dimensions etc, are grabbed from F1Technical.com, through a sophisitcated scraping script that I developed.'''
    st.markdown(f"<span style='font-family:optima;'>{content0}</span>",unsafe_allow_html=True)
    st.markdown(f"<span style='font-family:optima;'>{content1}</span>",unsafe_allow_html=True)

    st.markdown("> <span style='font-family:Syne;'>Data</span>",unsafe_allow_html=True)
    content = '''Majority of the Data is sourced using FastF1 Library and some aspects of the dashboard directly collects data from the Eargast API independent on FastF1 as the mediatory source. '''
    st.markdown(f"<span style='font-family:optima;'>{content}</span>",unsafe_allow_html=True)

    