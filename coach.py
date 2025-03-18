from typing import Iterator
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field
from agno.utils.pprint import pprint_run_response
from agno.utils.log import logger
from tools.pdfWriter import write_pdf
from tools.csvWriter import write_csv

import json

class Analysis(BaseModel):
    summary: str = Field(..., description="Full analysis")

class Feedback(BaseModel):
    feedback: str = Field(..., description="Full feedback in markdown")
    pdf_path: str = Field(..., description="Path to the pdf")


class Coach(Workflow):
    
    techniqueAnalyzer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="You are an experienced olympic weightlifting coach asked to analyze a user's weightlifting technique",
        instructions=["You will be provided with pose estimate data", 
                      "Use the pose estimate data to generate a detailed analysis of the user's weightlifting technique",
                      "identifying errors and areas for improvement",
                      "Your feedback should be structured in markdown format.",
                      "Don't provide any actionable feedback or recommendations, just the analysis."],
        response_model=Analysis,
        structured_outputs=True,
        debug_mode=False,
        markdown=True,
    )

    feedbackGenerator: Agent = Agent(
        tools=[write_pdf],
        model=OpenAIChat(id="gpt-4o"),
        instructions=["You are an experienced olympic weightlifting coach.",
                      "You will be provided with the analysis of the user's weightlifting technique",
                      "Provide actionable feedback to the user based on the analysis",
                      "You also provide specific recommendations, drills, and exercises to address identified issues.",
                      "Your feedback should be detailed and technical.",
                      "Your feedback should be structured in markdown format.",
                      "Finally, end the with general encouragement and motivation to the user.",
                      "Generate the pdf of the feedback and return the path to the pdf."],
        markdown=True,
        response_model=Feedback,
        debug_mode=False,
    )

    programGenerator: Agent = Agent(
        tools=[write_csv],
        model=OpenAIChat(id="gpt-4o"),
        instructions=["You are an experienced olympic weightlifting coach.",
                      "You will be provided with the actionable feedback of the user's weightlifting technique",
                      "Provide a detailed 10 week, 3 days a week program for the user based on the actionable feedback",
                      "The program should include specific exercises, drills, and techniques to address identified issues.",
                      "The program should mention the number of sets, reps, percentage of weight and rest time for each exercise.",
                      "First half of the program should focus on correcting technique and form, while the second half should focus on strength and power.",
                      "The exercises should be grouped by day and week.",
                      "The exercises should be structured in a way that the user can easily follow the program.",
                      "The csv file should be structured in following format: Week, Day, Exercise, Sets, Reps, Weight, Rest Time",
                      "Write the program in csv file and return the path to the csv file.",
                      "Return the program in a structured csv format with each day's workout."],
        debug_mode=False,
    )

    def run(self, pose_estimate_data: dict,st)-> Iterator[RunResponse]:
        analysis_results = self.techniqueAnalyzer.run(json.dumps(pose_estimate_data))
        if analysis_results is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could complete the analysis, please try again later.",
            )
            return

        with st.expander("Technique Analysis", expanded=False):
            st.write(analysis_results.content.summary)
        
        retryCount=0
        while retryCount<3:
            feedback_results = self.feedbackGenerator.run(analysis_results.content.summary)
            if feedback_results is None or feedback_results.content.feedback is None or isinstance(feedback_results.content, str):
                retryCount+=1
                continue
            with st.expander("Actionable Feedback", expanded=False):
                st.write(feedback_results.content.feedback)
                st.write(f"Download the pdf of the feedback [here]({feedback_results.content.pdf_path})")
            break
        if retryCount>=3:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not generate feedback, please try again later.",
            )
            return
        yield (yield from self.programGenerator.run(feedback_results.content.feedback,stream=True))
        

