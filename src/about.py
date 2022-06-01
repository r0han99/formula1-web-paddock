import streamlit as st 



def about_cs():
    
   # font-base64
    with open('./assets/formula1-regular-base64.txt', 'r') as f:
        regular = f.read()

    with open('./assets/formula1-bold-base64.txt', 'r') as f:
        bold = f.read()

    with open('./assets/formula1-black-base64.txt', 'r') as f:
        black = f.read()

    # fonts-style preset
    style = '''
    <style>
    @font-face {
    font-family: 'formula1';'''+f'''
    src: url(data:application/x-font-woff;charset=utf-8;base64,{regular} )'''+''' format('woff');
    src: url(data:application/x-font-woff;charset=utf-8;base64,{black} )'''+''' format('woff');
    src: url(data:application/x-font-woff;charset=utf-8;base64,{bold} )'''+''' format('woff');
    }

    </style>
    '''
    st.markdown(f'{style}',unsafe_allow_html=True)

    
    st.markdown('''<center><h4 style="font-family:formula1">The About Page</h4></center>''',unsafe_allow_html=True)