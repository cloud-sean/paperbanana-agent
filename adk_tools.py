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
Tools for the PaperBanana ADK agent.
Thin wrappers around the existing PaperBanana pipeline code.
"""

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


async def extract_methodology_from_pdf(pdf_path: str) -> dict:
    """Extract the methodology/methods section and a figure caption from a PDF.

    Uses Gemini to read the PDF and pull out the content that PaperBanana
    needs to generate an academic diagram.

    Args:
        pdf_path: Absolute path to the PDF file on disk.

    Returns:
        A dict with keys 'content' (extracted methodology text) and
        'caption' (a suggested figure caption).
    """
    from google import genai
    from google.genai import types
    from .utils.generation_utils import gemini_client

    if gemini_client is None:
        return {"error": "Gemini client not initialized. Check your Google API key."}

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return {"error": f"PDF not found: {pdf_path}"}

    pdf_bytes = pdf_file.read_bytes()

    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=(
                "You are an expert scientific reader. Read this academic paper and extract:\n\n"
                "1. **content**: The methodology / methods section of the paper. "
                "Include the key pipeline steps, data processing stages, and analytical methods described. "
                "If there is no explicit 'Methods' section, synthesize the methodological approach from the abstract and body. "
                "Be detailed — include sample sizes, model names, data sources, and processing steps.\n\n"
                "2. **caption**: Write a single descriptive caption for an overview diagram of this paper's methodology. "
                "The caption should describe what the diagram should show (e.g., 'Overview of the XYZ pipeline showing ...').\n\n"
                "Return your answer as exactly two sections, using these exact headers:\n"
                "CONTENT:\n<the extracted methodology>\n\n"
                "CAPTION:\n<the suggested caption>"
            )),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8000,
        ),
    )

    text = response.text
    content = ""
    caption = ""

    if "CONTENT:" in text and "CAPTION:" in text:
        parts = text.split("CAPTION:")
        content = parts[0].replace("CONTENT:", "").strip()
        caption = parts[1].strip()
    else:
        # Fallback: use entire response as content
        content = text
        caption = "Overview of the paper's methodology and analytical pipeline."

    return {"content": content, "caption": caption}


async def generate_diagram(
    content: str,
    caption: str,
    aspect_ratio: str = "16:9",
    max_critic_rounds: int = 2,
    num_candidates: int = 1,
    exp_mode: str = "demo_planner_critic",
    tool_context=None,
) -> dict:
    """Generate an academic diagram using the PaperBanana pipeline.

    Calls the existing PaperBanana agents (Planner, Visualizer, Critic)
    to produce a publication-quality illustration.

    Args:
        content: The methodology text to visualize.
        caption: Figure caption describing what the diagram should show.
        aspect_ratio: Output aspect ratio, one of '16:9', '21:9', '3:2'.
        max_critic_rounds: Number of critic refinement rounds (0-3).
        num_candidates: Number of parallel candidates to generate.
        exp_mode: Pipeline mode — 'demo_planner_critic' or 'demo_full'.

    Returns:
        A dict with 'image_paths' (list of saved PNG paths) and 'status'.
    """
    from .agents.planner_agent import PlannerAgent
    from .agents.visualizer_agent import VisualizerAgent
    from .agents.stylist_agent import StylistAgent
    from .agents.critic_agent import CriticAgent
    from .agents.retriever_agent import RetrieverAgent
    from .agents.vanilla_agent import VanillaAgent
    from .agents.polish_agent import PolishAgent
    from .utils import config
    from .utils.paperviz_processor import PaperVizProcessor
    from .skill.run import extract_final_image_b64

    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting="auto",
        max_critic_rounds=max_critic_rounds,
        work_dir=PROJECT_ROOT,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    data_list = []
    for i in range(num_candidates):
        data_list.append({
            "filename": f"adk_candidate_{i}",
            "caption": caption,
            "content": content,
            "visual_intent": caption,
            "additional_info": {"rounded_ratio": aspect_ratio},
            "max_critic_rounds": max_critic_rounds,
        })

    results = []
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=num_candidates, do_eval=False
    ):
        results.append(result_data)

    if not results:
        return {"error": "Pipeline returned no results.", "image_paths": []}

    from PIL import Image
    from google.genai import types as genai_types

    output_dir = PROJECT_ROOT / "results" / "adk_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for idx, result in enumerate(results):
        # Log available keys for debugging
        b64_keys = [k for k in result.keys() if 'base64' in k or 'b64' in k]
        logger.warning(f"[ADK] Result keys with base64: {b64_keys}")
        logger.warning(f"[ADK] All result keys: {list(result.keys())}")

        b64 = extract_final_image_b64(result, exp_mode)
        if not b64:
            logger.warning(f"[ADK] extract_final_image_b64 returned empty for result {idx}")
            continue
        else:
            logger.warning(f"[ADK] Got b64 image, length={len(b64)}")
        if "," in b64:
            b64 = b64.split(",")[1]
        image_data = base64.b64decode(b64)
        img = Image.open(BytesIO(image_data))

        import time
        ts = time.strftime("%m%d_%H%M%S")
        save_path = output_dir / f"diagram_{ts}_{idx}.png"
        img.save(str(save_path), format="PNG")
        saved_paths.append(str(save_path))

        # Save as ADK artifact so the image renders in the web UI
        if tool_context:
            artifact = genai_types.Part.from_bytes(
                data=image_data,
                mime_type="image/png",
            )
            await tool_context.save_artifact(
                filename=f"diagram_{ts}_{idx}.png",
                artifact=artifact,
            )

    return {
        "image_paths": saved_paths,
        "status": f"Generated {len(saved_paths)} diagram(s). Images saved as artifacts.",
    }
