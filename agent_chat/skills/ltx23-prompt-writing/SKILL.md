---
name: ltx23-prompt-writing
description: Write LTX 2.3 video prompts (Lightricks) for text-to-video and image-to-video in Morpheus (Playground, Dawnforge). Use whenever the user asks for an LTX / LTX 2.3 prompt, or for a video prompt from a start image without naming another model.
---

# LTX 2.3 Prompt Writing

LTX reads one flowing paragraph of plain description, like a shot list written as prose.
It renders video **and sound**, so describe what is heard too.

## Workflow

1. Decide the mode:
   - **I2V** (a start image is attached): the image is frame one. Do not re-describe it in full.
     Name the subject and setting briefly so the model anchors, then spend the words on **what moves and changes**.
   - **T2V** (no image): describe subject, look and setting first, then the action.
2. Fix the length. Default **5 seconds** unless the user says otherwise. One clear action fits 5 s;
   do not cram a story into it. Up to ~10 s: at most two beats.
3. Write the paragraph in this order:
   1. The main action, in the first sentence.
   2. The subject: specific movements and gestures, expression changes. Body parts move, not "the scene".
   3. Appearance that matters for motion (hair, cloth, light on skin) — I2V: only what the image already shows.
   4. Environment and background motion (wind, crowd, water, particles).
   5. Camera: one move, named plainly — static, slow push-in, pan left, handheld follow, orbit, crane up.
   6. Light and colour, as they look on camera (golden hour backlight, neon reflections on wet asphalt).
   7. Sound: ambience, effects, and any speech in quotes with who says it.
4. Check against the rules below, then output the prompt in one ```prompt block.

## Rules

- One paragraph, present tense, **120–200 words**. No headings, no bullet points, no tags lists.
- Concrete over abstract: "her shoulders drop as she exhales" beats "she looks relieved".
  Avoid "cinematic", "beautiful", "epic", "4k", "masterpiece" — they do nothing.
- Motion must be physically plausible and continuous; no cuts, no scene changes inside one clip.
- Never contradict the start image (clothes, colours, count of people, time of day).
- No negative phrasing ("no blur") — LTX in Morpheus runs at cfg 1, so negatives are ignored anyway.
- Dialogue: short lines in quotes, attributed ("the woman says quietly, \"...\""), original language.

## Example (I2V, 5 s)

```prompt
The young woman at the rain-streaked café window slowly turns her head toward the camera and a small smile
spreads across her face. She lifts the white ceramic cup with both hands, steam curling past her chin, and takes
a careful sip while her eyes stay on the lens. Behind her, blurred figures with umbrellas hurry along the
street and a tram glides past, its headlights smearing across the wet glass. Raindrops run down the window
in thin crooked lines. The camera holds a medium close-up and pushes in very slowly. Soft grey daylight from
the window lights one side of her face, warm amber lamps glow in the café behind. The sound of steady rain
on glass, a muffled tram bell, the low murmur of the café and the clink of the cup on its saucer.
```
