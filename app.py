from typing import Iterator,Iterable, Union
import streamlit as st
import tools.wlAnalysis as wla
import tempfile
from coach import Coach
from agno.workflow import RunResponse
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv
load_dotenv()


st.title('Welcome to Weightlifting Analysis')
st.write('This app will help you analyze your weightlifting form.')
st.write('Please upload a video of your weightlifting form.')

f = st.file_uploader('Upload a video', type=['mp4', 'mov', 'avi'])
stframe = st.empty()
if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())

    analysis_results = None
    full_output = ""
    with(st.spinner('Analyzing your video...')):
        analysis_results = wla.analyze_weightLifting_video(tfile.name, 1,stframe)

    with(st.spinner('Consulting experts...')):
        coaching = Coach()
        response : Union[RunResponse, Iterable[RunResponse]] = coaching.run(pose_estimate_data=analysis_results,st=st)
        for resp in response:
            if isinstance(resp, RunResponse) and isinstance(resp.content, str):
                full_output += resp.content
        with st.expander("Workout program", expanded=True):
            st.markdown(full_output)


    st.success('Analysis complete!')
    tfile.close()
