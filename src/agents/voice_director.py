from crewai import Agent, LLM, Task

def voice_director(llm: LLM) -> Agent:
    return Agent(
        role="Audiobook Voice Director",
        goal="""
        Convert structured emotional segments into SSML.
        You DO NOT infer emotion; you execute it literally.
        Handle multiple characters, transitions, and prosody.
        """,
        backstory="""
        You are a professional audiobook director.
        Responsibilities:
        - Map VAD → SSML prosody (rate, pitch, volume)
        - Respect emotional intensity and dominance
        - Use pauses according to transitions:
            soft: 200-300ms
            sharp: 400-700ms
            break: 800ms+
        - Emphasis:
            Only if intensity >= 0.7 and dominance >= 0.4
            Apply sparingly: 1–2 phrases per sentence
        - Handle multiple characters by switching voice tags
        - Include micro-pauses to simulate breathing
        - Output well-formed SSML for XTTS
        - No explanations or extra text
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def ssml_task(agent: Agent, text: str) -> Task:
    """
    Task 2: Convert text + Mood Map into SSML
    
    The Voice Director uses the Mood Map to create SSML with:
    - <prosody> tags for rate and pitch control
    - <break> tags for dramatic pauses
    - <emphasis> tags for key phrases
    """
    return Task(
        description=f"""Using the Mood Map from the Emotional Analyst, add SSML markup to this text.

*** CRITICAL: TEXT PRESERVATION RULES - READ CAREFULLY ***

You MUST include EVERY SINGLE WORD from the original text below, starting from the VERY FIRST WORD.
The text may start mid-sentence - this is INTENTIONAL. Do NOT skip it.

FORBIDDEN ACTIONS (WILL CAUSE FAILURE):
- Skipping ANY words, even if they seem incomplete
- Changing, correcting, or "fixing" any words
- Paraphrasing or summarizing
- Removing sentences or phrases for any reason
- Adding new content not in the original
- Reordering sentences

The output MUST start with the same words as the original text starts with.
Every line of the original MUST appear in your SSML output.

=== ORIGINAL TEXT (include EVERY word, starting from "{text.split()[0] if text.split() else ''}") ===
{text}
=== END OF ORIGINAL TEXT ===

SSML Tags to use:
- <prosody rate="x-slow|slow|medium|fast|x-fast" pitch="x-low|low|medium|high|x-high">text</prosody>
- <break time="200ms|500ms|1s|2s"/>
- <emphasis level="reduced|moderate|strong">text</emphasis>

Guidelines:
1. Use <break/> tags for dramatic pauses between thoughts or at emotional beats
2. Adjust rate based on pacing (slow=contemplative, fast=urgent)
3. Adjust pitch based on emotion (high=excited, low=somber)
4. Use <emphasis> SPARINGLY - only for 1-2 KEY phrases per sentence maximum
5. Most text should have NO emphasis tags - use prosody for general emotion instead
6. Wrap everything in a <speak> root element

Return ONLY the complete SSML document with ALL original text preserved.""",
        agent=agent,
        expected_output="A complete, well-formatted SSML document with EVERY word from the original text preserved verbatim, enhanced with prosody, breaks, and emphasis tags"
    )

