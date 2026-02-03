from crewai import Agent, LLM, Task

def emotional_analyst(llm: LLM) -> Agent:
    return Agent(
        role="Narrative Emotion Annotator",
        goal="""
        Convert narrative text into a structured emotional timeline using
        Plutchik emotions and Valence–Arousal–Dominance (VAD) scores.
        The output must be suitable for expressive text-to-speech synthesis.
        """,
        backstory="""
        You are an expert in narrative psychology and affective computing.

        You MUST:
        - Classify emotions ONLY using Plutchik's 8 core emotions:
          joy, trust, fear, surprise, sadness, disgust, anger, anticipation.
        - Assign an intensity value between 0.0 and 1.0 for each emotion.
        - Assign VAD values:
            valence:   -1.0 (very negative) to 1.0 (very positive)
            arousal:    0.0 (calm) to 1.0 (excited)
            dominance:  0.0 (submissive) to 1.0 (dominant)

        Think like a voice director:
        - How should the line sound?
        - Where does emotion shift?
        - What should the listener feel?

        DO NOT write literary analysis.
        DO NOT explain your reasoning.
        Output ONLY structured data.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def analysis_task(agent: Agent, text: str) -> Task:
    """
    Task 1: Analyze text and produce a Mood Map
    
    The Emotional Analyst creates a detailed mood map with:
    - Emotional tone for each segment
    - Intensity level (1-10)
    - Pacing notes (fast, medium, slow)
    - Key phrases that need emphasis
    """
    return Task(
        description=f"""Analyze the following text for emotional content and create a Mood Map.
        
Text to analyze:
{text}

Available emotions: excited, sad, calm, tense, humorous, neutral, dramatic, hopeful, desperate

Create a detailed Mood Map with the following structure for each segment (1-3 sentences):
1. Text segment (exact quotes from the original)
2. Dominant emotion
3. Intensity (1-10 scale)
4. Suggested pacing (fast/medium/slow)
5. Key phrases that need emphasis
6. Any dramatic pauses needed

Format your response as a numbered list of segments with clear labels for each field.""",
        agent=agent,
        expected_output="A detailed Mood Map breaking down the text into emotional segments with intensity, pacing, and emphasis notes"
    )

