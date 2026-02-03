from crewai import Agent, LLM, Task

def styletts_interpreter(llm: LLM) -> Agent:
    return Agent(
        role="StyleTTS2 Parameter Specialist",
        goal="Convert SSML-tagged text into precise StyleTTS2 parameter JSON segments.",
        backstory="""
        You are an expert audio engineer specializing in the StyleTTS2 text-to-speech engine.
        Your job is to take SSML-marked text (which indicates emotion, prosody, and pauses)
        and convert it into a machine-readable JSON format that the StyleTTS2 engine can strictly follow.

        You understand the relationship between human emotions and the 3 key StyleTTS2 parameters:
        1. Alpha (timbre/speaker identity): Lower (0.1) = closer to target speaker. Higher (0.8) = more styled/deviant.
        2. Beta (prosody strength): Higher (0.8-0.9) = strong emotional expression. Lower (0.4) = flatter.
        3. Diffusion Steps: Higher (10-15) = higher quality/complex emotion. Lower (3-5) = standard/neutral.

        Input SSML signals:
        - <break/> tags: MUST become separate silence segments or start new text segments.
        - <prosody>: Maps directly to alpha/beta adjustments.
        - <emphasis>: Increases beta and diffusion steps.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def style_parameters_task(agent: Agent) -> Task:
    return Task(
        description="""
        Analyze the SSML text provided by the previous agent (SSML Critic) and convert it into a JSON list of audio segments for StyleTTS2.

        GUIDELINES:
        1. Break the text into natural segments based on <break/> tags and sentence boundaries.
        2. Infer the emotion for each segment based on the SSML tags (prosody, pitch, rate).
        3. Assign specific StyleTTS2 parameters (alpha, beta, diffusion_steps) for each segment.
        4. "text" field must contain the plain text (NO TAGS) for that segment.
        
        PARAMETER MAPPING GUIDE:
        - Neutral/Narration: alpha=0.3, beta=0.7, steps=5
        - Excited/Fast/High Pitch: alpha=0.1, beta=0.9, steps=10
        - Sad/Slow/Low Pitch: alpha=0.5, beta=0.5, steps=8
        - Tense/Whisper: alpha=0.2, beta=0.8, steps=10
        - Dramatic/Emphasis: alpha=0.15, beta=0.85, steps=12

        OUTPUT FORMAT:
        Return ONLY a raw JSON list of objects. Do not wrap in markdown or code blocks.
        Example:
        [
            {{
                "text": "Hello world.",
                "emotion": "neutral",
                "alpha": 0.3,
                "beta": 0.7,
                "diffusion_steps": 5
            }},
            {{
                "text": "I can't believe it!",
                "emotion": "excited",
                "alpha": 0.1,
                "beta": 0.9,
                "diffusion_steps": 10
            }}
        ]
        """,
        agent=agent,
        expected_output="A valid JSON list of segment objects containing text and StyleTTS2 parameters."
    )
