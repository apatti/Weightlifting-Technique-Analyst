# Weightlifting Technique Analyst (WTA)

## Overview

The Weightlifting Technique Analyst (WTA) is a multi-agent application designed to analyze Olympic weightlifting techniques from user-uploaded videos. It provides detailed feedback, identifies areas for improvement.

## Features

* **Video Analysis:**
    * Upload and analyze videos of snatch and clean & jerk lifts.
    * Uses computer vision (pose estimation) to extract key joint movements.
    * Generates textual descriptions of movement patterns.
* **Multi-Agent System:**
    * **Technique Analyzer Agent:**
        * Retrieves relevant information from a knowledge base (guidelines, expert advice, biomechanics).
        * Generates detailed technique analysis using Retrieval-Augmented Generation (RAG).
    * **Feedback Generator Agent:**
        * Translates analysis into actionable feedback and personalized drills.
        * Provides feedback in various tones.
    * **RAGAS Evaluator Agent (Future work):**
        * Evaluates the quality of feedback using RAGAS metrics (faithfulness, relevancy).
        * Provides data to fine tune the embedding model.
    * **Embedding Model Fine-tuning (Future work):** 
        * Continuously improves retrieval accuracy based on RAGAS feedback.
* **User Interface:**
    * User-friendly interface for video uploads and feedback viewing.
    * Visualization of bar paths and joint angles.

## Technologies Used

* **Computer Vision:** MediaPipe
* **Large Language Models (LLMs):** GPT-4 through OpenAI
* **Multi-agent framework:** [Agno](https://github.com/agno-agi/agno)
* Python, and a web framework such as streamlit.

## Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone [repository URL]
    cd weightlifting-technique-analyst
    ```

2.  **Install Dependencies:**

    ```bash
    uv sync
    ```

3.  **Set Up LLM API Keys:**

    * Obtain API keys for your chosen LLM provider (OpenAI, Google Cloud).
    * Configure the API keys in the application's configuration file.

4.  **Run the Application:**

    ```bash
    uv run streamlit run app.py 
    ```

5.  **Access the Application:**

    * Open your web browser and navigate to the application's URL.

## Usage

1.  Upload a video of your weightlifting attempt.
2.  The application will process the video and generate an analysis report.
3.  View the analysis report, which includes feedback and recommendations.
4.  Use the feedback to improve your technique.

## Future Enhancements

1. Leverages RAGAS for evaluation.
2. Fine-tune an embedding model for enhanced accuracy.
3. Bar-bell detection to enhance pose analysis.
