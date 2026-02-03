from crewai import Agent, LLM, Task

def ssml_critic(llm: LLM) -> Agent:
    """
    Agent 3: The SSML Critic
    
    Role: SSML Quality Auditor who validates markup structure and
    checks for robotic or unnatural pacing.
    """
    return Agent(
        role="SSML XML Validator",
        goal="Validate SSML markup syntax and ensure proper XML structure",
        backstory="""You are a technical quality control specialist for SSML/XML markup.
        Your expertise is in validating XML syntax, not story content or creative writing.
        
        You check SSML documents for:
        - Proper tag closure and nesting
        - Valid attribute values
        - Appropriate use of prosody and breaks
        
        You NEVER analyze story content, provide writing tips, or suggest narrative changes.
        You ONLY validate XML markup structure.
        
        You output ONLY valid SSML - no explanations, no commentary, no analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def validation_task(agent: Agent, context: list[Task] = None) -> Task:
    """
    Task 3: Validate and critique the SSML
    
    The SSML Critic checks for:
    - Proper XML structure and tag nesting
    - Robotic patterns (too uniform, missing variation)
    - Missing emotional beats
    - Technical errors in attribute values
    """
    return Task(
        description="""CRITICAL: You are an SSML MARKUP validator with TWO responsibilities:

1. VALIDATE TEXT PRESERVATION (MOST IMPORTANT)
   - The SSML output must contain ALL words from the original text
   - No words should be changed, removed, or paraphrased
   - If text has been modified, this is a CRITICAL ERROR
   - If text is missing or altered, you MUST reject and note the issue

2. VALIDATE XML SYNTAX
   - All <prosody>, <break>, <emphasis> tags properly closed
   - Root <speak> element present
   - Valid attributes: rate (x-slow|slow|medium|fast|x-fast), pitch (x-low|low|medium|high|x-high), time (e.g., 500ms, 1s)
   - Emphasis used SPARINGLY (1-2 phrases per sentence max, not individual words)
   - If you see multiple <emphasis> tags on consecutive words, REMOVE them - this is over-tagging

DO NOT:
- Analyze story content
- Provide writing advice
- Suggest narrative improvements
- Rewrite or modify ANY text content

Output Format:
- If SSML is valid AND text is preserved: output it EXACTLY as received
- If SSML has XML errors: fix ONLY the XML tags, preserve all text
- If text was modified/removed: output the SSML but note that text integrity was compromised
- NEVER add explanations, analysis, or commentary
- NEVER include anything except the <speak>...</speak> XML document

Return ONLY valid SSML markup. Nothing else.""",
        agent=agent,
        expected_output="Valid SSML document with original text preserved and no explanations or commentary",
        context=context
    )

