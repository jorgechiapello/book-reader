---
description: How to use the IndexTTS-2 pipeline for emotionally expressive narration
---

# IndexTTS-2 Narration Workflow

This workflow uses the IndexTTS-2 engine via ComfyUI to generate highly expressive, duration-controlled audio for books and stories.

## Prerequisites

1. **ComfyUI**: Must be running at http://localhost:8188
2. **Custom Nodes**: Install `ComfyUI-Index-TTS` (by chenpipi0807) in your ComfyUI nodes folder.
3. **Reference Audio**: Ensure you have a speaker reference WAV file in your `voices/` directory.

## Steps

1. **Enter Task View**: Use `/indextts2` or call the workflow via CLI.
// turbo
2. **Run Workflow**:
   ```bash
   python src/main.py --engine indextts2 --text "Your story text here" --voice voices/Narrator.wav
   ```

## Key Features

- **Soft Instructions**: Unlike other engines, IndexTTS-2 uses natural language descriptions (e.g., "Whispering with grief") instead of numeric sliders.
- **Narrative Awareness**: The pipeline uses an Emotional Analyst and an IndexTTS-2 Interpreter to automatically map the story's emotional arc to the voice delivery.
- **Duration Control**: (Auto-managed) The engine ensures segments are timed naturally based on the text.

## Configuration

You can customize the output in `output/index_tts2/`. The pipeline generates:
- `.wav`: Final merged audio.
- `.ssml`: The intermediate emotional markup.
- `.moodmap`: A summary of the emotional arc of the chapter.
