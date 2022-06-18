import streamlit as st 
from pathlib import Path
import base64


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def about_cs():
    

    # fonts-style preset
     # font-style
    font_url = '''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap" rel="stylesheet">
    '''
    st.markdown(f"{font_url}",unsafe_allow_html=True)

    # fonts-style preset
    style = '''
    <style>
    html, body, [class*="css"] {
        font-family: syne; 
        
    }

    </style>
    '''
    st.markdown(style,unsafe_allow_html=True)

    
    st.markdown('''<center><h4 style="font-family:syne;font-weight:900;">Home Page</h4></center>''',unsafe_allow_html=True)
    st.markdown('***')
    st.sidebar.markdown('***')

    
    cols = st.columns([3.5,4])
    cols[0].image('./assets/f1.gif',caption='By Twain Forsythe')
    info = '''
    "Here is the portal directly taking you into the Team Paddocks giving you a deeper look into the Past Races, Strategies, Times Recorded and much more."
    '''
    cols[1].markdown(f'''<h3 style="font-family:syne">For the vehement Formula One fans,<br> <span style='font-size:22px;'>{info}</span></h3>''',unsafe_allow_html=True)
    cols[1].markdown(f'''<p style='font-family:syne;'; font-size:18px;><span style='font-weight:normal;'>Engineered by</span> <u style='font-size:22px; font-weight:900; color:orange;'>r0han</u> <a href='https://github.com/r0han99'> <img src='data:image/png;base64,{img_to_bytes('./assets/github.png')}' class='img-fluid' width=35> </a></p>''',unsafe_allow_html=True)
    cols[1].markdown(f'''<p style='font-family:syne;'; font-size:18px;><span style='font-weight:normal;'>Powered by</span><a href='http://streamlit.io/'> <u style='font-size:18px; font-weight:900;'>Streamlit</u></a> + <a href='https://theoehrly.github.io/Fast-F1/'><u style='font-size:18px; font-weight:900;'>Fast F1</u></a></p>''',unsafe_allow_html=True)

    

    st.markdown('***')
    st.markdown('''<center><span style="font-family:syne;font-weight:800; font-size:25px">Gallery</span></center>''',unsafe_allow_html=True)
    st.markdown('***')
    st.markdown(f'''<span style="font-family:syne; font-size:25px">Race Schedule <sub style='font-size:15px'>(<img src='data:image/png;base64,{img_to_bytes('./assets/checkered-flag.png')}' class='img-fluid' width=25> completed)</sub></span>''',unsafe_allow_html=True)
    

    # race schedule
    st.image('./assets/Home/s24.png')
    st.markdown('''<span style="font-family:syne; font-size:25px">Grand Prix Analysis - Session Selection</span>''',unsafe_allow_html=True)
    
    # session select - GP analysis
    cols = st.columns(2)
    cols[0].image('./assets/Home/s23.png',caption='session selection')
    cols[1].image('./assets/Home/s21.png',caption='categories')

    # summarise
    st.markdown('')
    st.markdown('''<span style="font-family:syne; font-size:25px">Qualifying - Summary</span>''',unsafe_allow_html=True)
    cols = st.columns(2)
    cols[0].image('./assets/Home/s20.png',caption='Knockouts')
    cols[1].image('./assets/Home/s19.png',caption='Full Grid')
    
    st.markdown('')
    st.markdown('''<span style="font-family:syne; font-size:25px">Qualifying - Nerd Mode</span>''',unsafe_allow_html=True)
    st.image('./assets/Home/s16.png')

    st.markdown('''<span style="font-family:syne; font-size:18px">Individual Driver Analysis</span>''',unsafe_allow_html=True)
    st.markdown('')
    cols = st.columns(3)
    cols[0].image('./assets/Home/s15.png')
    cols[1].image('./assets/Home/s14.png')
    cols[2].image('./assets/Home/s13.png')
    
    cols = st.columns(2)
    cols[0].markdown('')
    cols[0].markdown('')
    cols[0].markdown('')
    cols[0].image('./assets/Home/s12.png',)
    cols[1].markdown('')
    cols[1].image('./assets/Home/s11.png')
    
    st.markdown('''<span style="font-family:syne; font-size:18px">Comparative Driver Analysis</span>''',unsafe_allow_html=True)
    cols = st.columns(2)
    exp = cols[0].expander('Manually Selected Drivers')
    exp.image('./assets/Home/s10.png')
    exp = cols[1].expander('Team Drivers')
    exp.image('./assets/Home/s9.png')

    # race
    st.markdown('')
    st.markdown('''<span style="font-family:syne; font-size:25px">Race Analysis</span>''',unsafe_allow_html=True)

    cols = st.columns(2)
    cols[0].image('./assets/Home/s8.png',caption='Race Mode')
    cols[1].markdown('')
    cols[1].markdown('')
    cols[1].markdown('')
    cols[1].image('./assets/Home/s7.png',caption='Category-Driver Analysis')

    cols = st.columns([2,6,2])
    cols[1].image('./assets/Home/s6.png',caption='Race Strategy')

    cols = st.columns(2)
    cols[0].image('./assets/Home/s2.png',caption='Race Standings')
    cols[1].image('./assets/Home/s1.png',caption='Retirements')

    
    st.markdown('***')
    
