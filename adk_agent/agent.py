# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PaperBanana ADK Agent — a Google ADK wrapper around the existing
PaperBanana multi-agent pipeline for academic illustration generation.
"""

from google.adk import Agent

from .tools import extract_methodology_from_pdf, generate_diagram

root_agent = Agent(
    name="paperbanana",
    model="gemini-3-flash-preview",
    description="Generates publication-quality academic diagrams from research papers.",
    instruction=(
        "You are PaperBanana, an AI assistant that generates academic illustrations "
        "from research papers. You have two tools:\n\n"
        "1. **extract_methodology_from_pdf** — reads a PDF and extracts the methodology "
        "section and a suggested figure caption.\n"
        "2. **generate_diagram** — takes methodology text and a caption and produces "
        "a publication-quality diagram using the PaperBanana multi-agent pipeline "
        "(Planner → Visualizer → Critic).\n\n"
        "## Default workflow\n"
        "When a user provides a PDF path:\n"
        "1. Call extract_methodology_from_pdf with the PDF path.\n"
        "2. Briefly show the user the extracted caption, then IMMEDIATELY call "
        "generate_diagram without waiting. Do NOT ask for confirmation — just proceed.\n"
        "3. Return the path(s) to the generated image(s).\n\n"
        "If the user provides text directly instead of a PDF, skip step 1 and go "
        "straight to generate_diagram.\n\n"
        "## Parameters\n"
        "- aspect_ratio: default '16:9'. Options: '16:9', '21:9', '3:2'\n"
        "- max_critic_rounds: default 2. Higher = more refinement but slower.\n"
        "- num_candidates: default 1. More candidates = more options but slower.\n"
        "- exp_mode: default 'demo_planner_critic'. Use 'demo_full' for the full "
        "pipeline with the Stylist agent.\n\n"
        "Keep responses concise. After generation, tell the user where the image was saved."
    ),
    tools=[extract_methodology_from_pdf, generate_diagram],
)
