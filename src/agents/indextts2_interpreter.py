from crewai import Agent, LLM, Task

def indextts2_interpreter(llm: LLM) -> Agent:
    return Agent(
        role="IndexTTS-2 Emotional Director",
        goal="Convert SSML-tagged text into natural language 'soft instructions' for IndexTTS-2.",
        backstory="""
        You are an elite voice director specializing in expressive, emotionally-driven text-to-speech.
        You understand the nuance of human speech and how to describe it using natural language.
        
        IndexTTS-2 uses "soft instructions" to control emotion. Instead of setting technical parameters, 
        you describe how the voice should sound.
        
        Your job is to:
        1. Break the SSML text into natural segments.
        2. For each segment, provide the clean text (without tags).
        3. For each segment, write a concise but vivid "soft instruction" describing the emotional delivery.
        
        Examples of soft instructions:
        - "Whispering with a hint of fear and urgency."
        - "Energetic and joyful narration, very upbeat."
        - "A cold, monotonous, and menacing tone."
        - "Warm, comforting, and slow-paced storytelling."
        - "Frantic and breathless, as if running."
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def indextts2_task(agent: Agent, context: list[Task] = None) -> Task:
    return Task(
        description="""
        Analyze the SSML text provided (from SSML Critic) and convert it into a JSON list of segments for IndexTTS-2.
        
        GUIDELINES:
        1. Break the text into logical segments based on <break/> tags and sentence boundaries.
        2. "text" field must contain the plain text (NO SSML TAGS).
        3. "soft_instruction" field must contain a short (3-8 words) natural language description of the emotion and pacing.
        4. "role" field should be "Narrator" by default unless specified otherwise.
        5. "emotion" field is a single keyword summary (e.g., "sad", "joyful", "neutral").

        OUTPUT FORMAT:
        Return ONLY a raw JSON list of objects.
        Example:
        [
            {
                "text": "I can't believe you're actually here.",
                "soft_instruction": "Relieved and slightly breathless with surprise.",
                "emotion": "surprise",
                "role": "Narrator"
            },
            {
                "text": "It has been so long.",
                "soft_instruction": "Nostalgic and warm with a slow pace.",
                "emotion": "nostalgic",
                "role": "Narrator"
            }
        ]
        """,
        agent=agent,
        expected_output="A valid JSON list of segment objects with text and soft_instruction.",
        context=context
    )
