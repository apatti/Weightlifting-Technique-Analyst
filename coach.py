from typing import Iterator
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pydantic import BaseModel, Field
from agno.utils.pprint import pprint_run_response
from agno.utils.log import logger
import json

class Analysis(BaseModel):
    summary: str = Field(..., description="Full analysis")

class Coach(Workflow):
    
    techniqueAnalyzer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="You are an experienced olympic weightlifting coach asked to analyze a user's weightlifting technique",
        instructions=["You will be provided with pose estimate data", 
                      "Use the pose estimate data to generate a detailed analysis of the user's weightlifting technique",
                      "identifying errors and areas for improvement",
                      "Don't provide any actionable feedback or recommendations, just the analysis."],
        response_model=Analysis,
        structured_outputs=True,
    )

    feedbackGenerator: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=["You are an experienced olympic weightlifting coach.",
                      "You will be provided with the analysis of the user's weightlifting technique",
                      "Provide actionable feedback to the user based on the analysis",
                      "You also provide specific recommendations, drills, and exercises to address identified issues.",
                      "Your feedback should be detailed and technical.",
                      "Finally, end the with general encouragement and motivation to the user."],
        markdown=True,
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
        
        yield (yield from self.feedbackGenerator.run(analysis_results.content.summary,stream=True))
        

