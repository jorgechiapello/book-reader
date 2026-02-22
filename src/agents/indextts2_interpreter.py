from crewai import Agent, LLM, Task


def indextts2_interpreter(llm: LLM) -> Agent:
    return Agent(
        role="IndexTTS-2 Emotional Director",
        goal="Convert Mood Map analysis into IndexTTS-2 segment format with emo_vector.",
        backstory="""
        You are an elite voice director specializing in expressive, emotionally-driven text-to-speech.
        IndexTTS-2 uses an 8-value emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].

        Rules for emo_vector:
        - Each value must be between 0 and 1 (no value > 1).
        - The sum does NOT need to equal 1.
        - Calm/neutral segments: use low values (e.g. calm 0.3–0.5, others near 0).
        - Intense segments: dominant emotions can approach 0.8–1.0; others stay lower.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )


def indextts2_task(agent: Agent, context: list[Task] | None = None) -> Task:
    return Task(
        description="""
        Use the Mood Map from the Emotional Analyst to produce a JSON list of segments for IndexTTS-2.

        For each segment in the Mood Map:
        1. "text": plain text (no SSML tags).
        2. "emo_vector": array of exactly 8 floats [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
           - Each value 0–1. Calm segments use low values; intense segments can use 0.8–1.0 for dominant emotion.
        3. "role": "Narrator" unless otherwise specified.

        OUTPUT FORMAT:
        Return ONLY a raw JSON list. Example:
        [
            {"text": "It was a dark night.", "emo_vector": [0, 0, 0.15, 0.2, 0, 0.05, 0, 0.5], "role": "Narrator"},
            {"text": "She screamed.", "emo_vector": [0, 0.1, 0.1, 0.9, 0, 0, 0.3, 0], "role": "Narrator"}
        ]
        """,
        agent=agent,
        expected_output="A valid JSON list of segment objects with text, emo_vector (8 floats), and role.",
        context=context,
    )


def indextts2_retry_task(
    agent: Agent,
    invalid_segment: dict,
    feedback: str,
    context: list[Task] | None = None,
) -> Task:
    """Task to fix a segment with invalid emo_vector."""
    return Task(
        description=f"""
        The following segment has an invalid emo_vector.

        {feedback}

        Invalid segment: {invalid_segment}

        Fix the emo_vector and return ONLY a single-object JSON array with the corrected segment.
        Example: [{{"text": "...", "emo_vector": [0, 0, 0.1, 0.2, 0, 0.05, 0, 0.6], "role": "Narrator"}}]
        """,
        agent=agent,
        expected_output="A valid JSON list with one corrected segment object.",
        context=context,
    )
