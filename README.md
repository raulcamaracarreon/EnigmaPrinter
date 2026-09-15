<div align="center">

# EnigmaPrinter

### *In the Beginning was the Word.*

**Audio-first AI video production**

EnigmaPrinter turns narration into a structured visual timeline, plans the shots around the spoken story, generates or selects the media required for each one, and assembles the final video around the real timing of the voice.

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#development-status)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Derived from [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) by Harry and contributors.**

</div>

---

## From words to vision

EnigmaPrinter starts with the spoken story.

Instead of treating narration as one more asset added during editing, the narration becomes the **temporal authority** of the production. Its real duration and semantic structure can define the scene timeline, the shot plan, the generated media, and the final composition.

> **The word comes first. The images follow.**

```text
Script
  ↓
Narration
  ↓
Narration Alignment
  ↓
Semantic Scene Timeline
  ↓
Active Shot Plan
  ↓
Visual Planning
  ↓
Subject Anchors + Scene Prompts
  ↓
AI Images / AI Video / Other Media
  ↓
Timeline-Aware Composition
  ↓
Final Video
```

EnigmaPrinter keeps the standard MoneyPrinterTurbo workflow available. The audio-first timeline is an additional production path, not a replacement for the original workflow.

---

## What EnigmaPrinter changes

MoneyPrinterTurbo is an all-in-one short-video generator that can go from a topic or script to voiceover, footage, subtitles, music, and final composition.

EnigmaPrinter develops a different production philosophy on top of that foundation: **narration first, visual planning second, media generation third**.

Its current development path focuses on:

- **audio-first timing** based on the real narration duration
- **semantic scene planning** instead of arbitrary clip segmentation
- a human-reviewable **Active Shot Plan** before expensive generation begins
- **visual continuity** through reusable Subject Anchors
- **AI image and AI video generation** as native visual sources
- **hybrid media planning**, allowing different shots to use different providers or media types
- **timeline-aware composition** using the exact planned start, end, and duration of each shot
- compatibility with **local AI workflows**, including Ollama and ComfyUI

The goal is not to redesign MoneyPrinterTurbo for its own sake. EnigmaPrinter is being developed **production-first**: real videos expose friction, and the tool evolves where that friction matters.

---

## Core workflow

### 1. Audio-first timeline

The final narration can determine the visual structure of the project.

The resulting timeline preserves:

- exact shot start and end times
- exact shot duration
- scene order
- narration synchronization
- validated shot-length constraints

Once the timeline is locked, later planning stages cannot silently change its timing, shot count, or order.

### 2. Active Shot Plan

The visual structure is managed through an **Active Shot Plan**:

```text
Base Media Shot Plan
        ↓
Optional AI Refine
        ↓
Active Shot Plan
        ↓
Generate Visual Prompts
        ↓
Lock Timeline
```

This creates a review point before image or video generation begins.

### 3. Visual continuity

Recurring subjects can be represented with reusable anchors:

```text
[CHAR_n]      Characters
[LOC_n]       Locations
[OBJ_n]       Objects
[VEH_n]       Vehicles
[CREATURE_n] Creatures
```

Scene prompts reference these anchors instead of independently reinventing the same character, location, object, vehicle, or creature in every shot.

### 4. AI media generation

Current EnigmaPrinter development includes tested workflows for:

- **ComfyUI text-to-image**
- **OpenAI-compatible image generation**
- **ComfyUI video generation**
- local ComfyUI workflows
- AI-generated visual prompts
- Subject Anchors + Scene Prompts
- still-image animation

A ComfyUI workflow for **LTX-2.5 text-to-video** is included in the repository.

### 5. Hybrid Media Plan

Not every shot needs the same kind of media.

EnigmaPrinter supports two strategies.

**Single Source** uses one selected visual provider for all shots.

```text
Shot 1 → ComfyUI T2I
Shot 2 → ComfyUI T2I
Shot 3 → ComfyUI T2I
```

**Media Plan (Mixed)** allows each shot to select its own strategy.

```text
Shot 1 → OpenAI Image
Shot 2 → ComfyUI T2I
Shot 3 → ComfyUI Video
Shot 4 → OpenAI Image
Shot 5 → ComfyUI Video
```

This makes it possible to reserve more expensive video generation for shots that genuinely benefit from motion while using still images elsewhere.

### 6. Timeline-aware composition

When the advanced timeline is active, media is composed according to its exact planned window.

The compositor respects:

- shot start time
- shot end time
- shot duration
- narration timing
- image/video media type
- still-image motion settings

The legacy compositor remains available for standard workflows.

---

## Production modes

### Standard / Legacy

The original MoneyPrinterTurbo-style workflow remains available.

```text
Script → Media → Voice → Subtitles → Composition
```

No EnigmaPrinter shot timeline is required.

### Timeline + Single Provider

One provider generates the visual material for a locked audio-first timeline.

```text
Narration
   ↓
Shot Timeline
   ↓
Visual Prompts
   ↓
Single AI Provider
   ↓
Timeline Composition
```

### Timeline + Hybrid Media

The full EnigmaPrinter workflow can choose media per shot.

```text
Narration
   ↓
Semantic Timeline
   ↓
Active Shot Plan
   ↓
Visual Continuity
   ↓
Per-Shot Media Plan
   ↓
Images + Video
   ↓
Timeline Composition
```

---

## Still-image motion

Generated still images currently support:

- **Static**
- **Slow Zoom In**

These modes work with standard production and with the timeline-aware workflow where applicable.

---

## Local AI friendly

EnigmaPrinter is designed to use local AI components where practical.

Current development and testing include:

- **Ollama** for local utility-LLM tasks
- **ComfyUI** for local image generation
- **ComfyUI** for local video generation

Cloud APIs can still be used when they provide better quality, speed, model access, or convenience.

---

## Relationship to MoneyPrinterTurbo

EnigmaPrinter would not exist without [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo).

MoneyPrinterTurbo, created by **Harry and contributors**, provides the original application architecture, WebUI, API, provider integrations, media pipeline, subtitle and voice systems, video composition foundation, and many other capabilities inherited by this project.

EnigmaPrinter is a derivative project with an independent development path focused primarily on audio-first planning, semantic timelines, AI-generated media, visual continuity, hybrid shot strategies, and timeline-aware composition.

Where a change is generic and useful beyond EnigmaPrinter, it may be evaluated separately for possible contribution back to the upstream MoneyPrinterTurbo project.

### Upstream

- Original project: [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)
- Original author: **Harry**
- License: **MIT**

The original license and copyright notice are preserved in this repository.

---

## Development status

EnigmaPrinter is currently an **active development fork**.

The audio-first planning and AI-media workflow is functional and covered by regression tests, but independent packaging and release documentation are still being prepared.

For now, this repository should be treated as a **source/development release** rather than as a finished standalone distribution.

The current development loop is intentionally simple:

```text
Produce videos
      ↓
Find real workflow friction
      ↓
Improve the tool
      ↓
Produce more videos
      ↓
Measure results
      ↓
Repeat
```

---

## Development checkout

Clone this repository if you specifically want the EnigmaPrinter development branch and its audio-first, visual-planning, and hybrid-media features:

```bash
git clone https://github.com/raulcamaracarreon/EnigmaPrinter.git
cd EnigmaPrinter
```

Python **3.11+** is required by the inherited MoneyPrinterTurbo codebase.

Independent EnigmaPrinter installation and packaging instructions will be added after the standalone setup path has been validated. Until then, upstream MoneyPrinterTurbo installation documentation can be useful for understanding the underlying application architecture, but upstream binary releases do **not** contain EnigmaPrinter-specific features.

---

## License and attribution

EnigmaPrinter is distributed under the **MIT License**.

It is a derivative work of [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo).

Original copyright notice:

```text
Copyright (c) 2024 Harry
```

The original MIT license and copyright notice are preserved in [`LICENSE`](LICENSE).

---

## Contributing

EnigmaPrinter is evolving through real production use.

Bug reports, workflow observations, compatibility improvements, and clearly scoped contributions are welcome. Generic improvements that may also benefit MoneyPrinterTurbo can be considered separately for upstream contribution.

---

<div align="center">

# EnigmaPrinter

### *In the Beginning was the Word.*

**The word comes first. The images follow.**

</div>
