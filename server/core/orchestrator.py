"""
server/core/orchestrator.py

Unified Autonomous Content Creation Orchestrator that coordinates:
Strategy -> Trend Discovery -> Deep Research -> Storyboard & Scriptwriting ->
Technical & Creative QA -> Editing Timeline -> Export -> Learning.
"""

import os
import re
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List

from server.core.agent_system import (
    StrategyAgent, TrendDiscoveryAgent, ResearchAgent, 
    CreativeDirectorAgent, QualityAssuranceAgent, LearningAgent,
    VideoScriptPlan
)
from server.core.queue import update_job, add_log, save_stage_checkpoint, get_stage_checkpoint, get_job, manager
from config import load_config

class AutonomousStudioOrchestrator:
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id
        self.strategy_agent = StrategyAgent()
        self.trend_agent = TrendDiscoveryAgent()
        self.research_agent = ResearchAgent()
        self.director_agent = CreativeDirectorAgent()
        self.qa_agent = QualityAssuranceAgent()
        self.learning_agent = LearningAgent()

    def _log(self, msg: str):
        logging.info(f"[Orchestrator:{self.job_id}] {msg}")
        if self.job_id:
            add_log(self.job_id, msg)

    def _update_progress(self, step: str, progress: float):
        if self.job_id:
            update_job(self.job_id, step=step, progress=progress)

    def produce_content_pipeline(
        self, 
        topic: str, 
        duration_sec: int = 45, 
        style: str = "Cinematic Documentary", 
        region: str = "Global",
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end intelligent creative pipeline with checkpoints and validation.
        """
        self._update_progress("Synthesizing Strategy & Preferences", 0.05)
        self._log(f"Starting Autonomous Creative Production for topic: '{topic}'")

        # 1. Learning Check: Inject saved user preferences
        preferred_style = self.learning_agent.get_preference("user_preference", "default_style", style)
        self._log(f"Loaded active style profile: '{preferred_style}'")

        # 2. Research Phase
        self._update_progress("Deep Research & Fact Verification", 0.20)
        research = self.research_agent.research_topic(topic)
        self._log(f"Research complete: {len(research.key_insights)} insights, {len(research.facts_ledger)} verified facts.")
        if self.job_id:
            save_stage_checkpoint(self.job_id, "research", research.model_dump())

        # 3. Creative Storyboard & Scriptwriting
        self._update_progress("Directing Scenes & Writing Script", 0.45)
        script_plan = self.director_agent.create_storyboard_plan(
            topic=topic,
            research=research,
            target_duration_sec=duration_sec,
            style=preferred_style
        )
        self._log(f"Script planned: '{script_plan.title}' with {len(script_plan.scenes)} scenes.")

        # 4. Quality Assurance Review & Autonomous Self-Healing Retry Loop
        self._update_progress("Quality Assurance Inspection", 0.65)
        qa_result = self.qa_agent.review_plan(script_plan, research)
        self._log(f"Initial QA Score: {qa_result.score}/100. Passed: {qa_result.passed}")

        # Self-healing loop: if QA score < 80, automatically critique and rewrite
        max_retries = 2
        retry_count = 0
        while (not qa_result.passed or qa_result.score < 80) and retry_count < max_retries:
            retry_count += 1
            critique_feedback = (
                f"Issues to fix:\n"
                f"- Defects: {', '.join(qa_result.technical_defects) if qa_result.technical_defects else 'None'}\n"
                f"- Improvements needed: {', '.join(qa_result.creative_improvements)}\n"
                f"- Recommendations: {', '.join(qa_result.recommended_repairs)}"
            )
            self._update_progress(f"Autonomous Script Revision (Pass {retry_count})", 0.50 + (retry_count * 0.08))
            self._log(f"QA threshold not met ({qa_result.score}/100). Triggering autonomous revision pass {retry_count}/{max_retries}...")
            
            # Preserve any scenes marked locked
            locked = [s for s in script_plan.scenes if s.locked]
            script_plan = self.director_agent.create_storyboard_plan(
                topic=topic,
                research=research,
                target_duration_sec=duration_sec,
                style=preferred_style,
                feedback=critique_feedback,
                locked_scenes=locked
            )
            qa_result = self.qa_agent.review_plan(script_plan, research)
            self._log(f"Revision pass {retry_count} QA Score: {qa_result.score}/100. Passed: {qa_result.passed}")

        if self.job_id:
            save_stage_checkpoint(self.job_id, "storyboard", script_plan.model_dump())
            save_stage_checkpoint(self.job_id, "qa_review", qa_result.model_dump())

        # 5. Output Packaging
        output_data = {
            "topic": topic,
            "title": script_plan.title,
            "hook": script_plan.hook,
            "duration": script_plan.target_duration_sec,
            "style": preferred_style,
            "scenes": [s.model_dump() for s in script_plan.scenes],
            "research_summary": research.summary,
            "qa_score": qa_result.score,
            "qa_passed": qa_result.passed,
            "status": "ready_for_render" if auto_approve else "awaiting_approval"
        }

        if auto_approve:
            self._update_progress("Production Approved & Ready", 1.0)
            if self.job_id:
                update_job(self.job_id, status="completed", progress=1.0, result=json.dumps(output_data))
        else:
            self._update_progress("Awaiting User Approval", 0.75)
            if self.job_id:
                update_job(self.job_id, status="awaiting_approval", progress=0.75, result=json.dumps(output_data))

        # Store creative decision in memory
        self.learning_agent.record_preference("creative_memory", f"topic_{topic[:30]}", {
            "title": script_plan.title,
            "hook": script_plan.hook,
            "score": qa_result.score
        })

        self._log("Autonomous Orchestrator completed planning stage successfully.")
        return output_data

    def render_approved_storyboard(self, scenes_override: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Synthesizes narration, visual frames/B-roll, and renders the approved Storyboard plan
        into a master MP4 video file inside workspace/plan_<job_id>/master_render.mp4.
        """
        if not self.job_id:
            raise ValueError("job_id is required to render storyboard.")

        storyboard_data = get_stage_checkpoint(self.job_id, "storyboard")
        if not storyboard_data:
            job = get_job(self.job_id)
            if job and job.get("result"):
                try:
                    storyboard_data = json.loads(job["result"])
                except Exception:
                    pass
        if not storyboard_data:
            raise ValueError(f"Storyboard checkpoint not found for job {self.job_id}")

        if scenes_override:
            storyboard_data["scenes"] = scenes_override
            save_stage_checkpoint(self.job_id, "storyboard", storyboard_data)

        scenes = storyboard_data.get("scenes", [])
        if not scenes:
            raise ValueError("Storyboard has no scenes to render.")

        title = storyboard_data.get("title", "Autonomous Studio Production")
        safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', self.job_id[:12])
        output_dir = os.path.join("workspace", f"plan_{safe_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Save storyboard & SEO metadata for History and Publish Hub
        with open(os.path.join(output_dir, "storyboard.json"), "w", encoding="utf-8") as f:
            json.dump(storyboard_data, f, indent=2)
        with open(os.path.join(output_dir, "seo_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": title,
                "description": storyboard_data.get("research_summary") or storyboard_data.get("hook", ""),
                "tags": ["ai studio", "autonomous director", storyboard_data.get("style", "documentary")]
            }, f, indent=2)

        config = load_config()
        from PIL import Image, ImageDraw, ImageFont
        from api_clients import PixabayClient, GoogleClient, WaveSpeedClient

        pixabay_key = config.get("PIXABAY_API_KEY", "")
        ws_key = config.get("WAVESPEED_API_KEY") or config.get("WAVESPEED_AI_KEY") or ""
        gemini_key = config.get("GEMINI_API_KEY", "")

        segment_files = []
        total_scenes = len(scenes)

        for idx, scene in enumerate(scenes):
            scene_num = scene.get("scene_number", idx + 1)
            dur = max(3.0, float(scene.get("duration_estimate_sec", 6.0)))
            narration = str(scene.get("narration", f"Scene {scene_num}")).strip()
            visual_desc = str(scene.get("visual_description", title)).strip()
            prog = 0.75 + (0.20 * (idx / max(1, total_scenes)))
            self._update_progress(f"Rendering Scene {scene_num}/{total_scenes}", round(prog, 2))
            self._log(f"Synthesizing media for Scene {scene_num}/{total_scenes}: {narration[:50]}...")

            # 1. Generate or synthesize audio for scene
            audio_path = os.path.join(output_dir, f"scene_{scene_num}.wav")
            audio_ok = False
            if gemini_key:
                try:
                    g_client = GoogleClient(config)
                    g_client.generate_tts(narration, audio_path, {"SPEAKER1": config.get("SPEAKER1", "Kore"), "CONTENT_STYLE": "Narration"})
                    audio_ok = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
                except Exception as e:
                    logging.warning(f"Gemini TTS fallback for scene {scene_num}: {e}")
            if not audio_ok and ws_key:
                try:
                    ws_client = WaveSpeedClient(ws_key)
                    ws_client.generate_speech(narration, "Brian", audio_path)
                    audio_ok = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
                except Exception as e:
                    logging.warning(f"WaveSpeed TTS fallback for scene {scene_num}: {e}")

            if not audio_ok:
                # Generate clean ambient audio track matching scene duration
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=220:duration={dur}",
                    "-af", "volume=0.03",
                    "-c:a", "pcm_s16le", audio_path
                ], check=True, capture_output=True)

            # 2. Generate visual frame / B-roll for scene
            seg_video_path = os.path.join(output_dir, f"scene_{scene_num}_seg.mp4")
            broll_path = os.path.join(output_dir, f"scene_{scene_num}_raw.mp4")
            has_broll = False

            if pixabay_key:
                try:
                    px = PixabayClient(pixabay_key)
                    query = " ".join(visual_desc.split()[:4]) or title
                    has_broll = px.download_video(query, broll_path, is_vertical=False)
                except Exception:
                    has_broll = False

            if has_broll and os.path.exists(broll_path):
                subprocess.run([
                    "ffmpeg", "-y",
                    "-stream_loop", "-1", "-i", broll_path,
                    "-i", audio_path,
                    "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=24",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", "-t", str(dur),
                    seg_video_path
                ], check=True, capture_output=True)
            else:
                # Create styled storyboard visual card using Pillow
                img_path = os.path.join(output_dir, f"scene_{scene_num}.png")
                img = Image.new("RGB", (1280, 720), color=(15, 18, 32))
                draw = ImageDraw.Draw(img)
                # Accent header bar
                draw.rectangle([(0, 0), (1280, 8)], fill=(16, 185, 129))
                draw.text((60, 50), f"SCENE {scene_num}  |  {title.upper()[:45]}", fill=(16, 185, 129))
                draw.text((60, 110), f"Visual: {visual_desc[:85]}", fill=(148, 163, 184))
                # Wrap narration text
                words = narration.split()
                lines = []
                cur = []
                for w in words:
                    cur.append(w)
                    if len(" ".join(cur)) > 56:
                        lines.append(" ".join(cur))
                        cur = []
                if cur:
                    lines.append(" ".join(cur))
                y_pos = 260
                for line in lines[:6]:
                    draw.text((60, y_pos), line, fill=(248, 250, 252))
                    y_pos += 42
                img.save(img_path)

                subprocess.run([
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", img_path,
                    "-i", audio_path,
                    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
                    "-pix_fmt", "yuv420p", "-vf", "scale=1280:720,fps=24",
                    "-c:a", "aac", "-shortest", "-t", str(dur),
                    seg_video_path
                ], check=True, capture_output=True)

            segment_files.append(seg_video_path)

        # 3. Concatenate all scene segments into master_render.mp4
        self._update_progress("Assembling Final Master Video", 0.95)
        self._log("Concatenating rendered scenes into master MP4...")
        concat_list_path = os.path.join(output_dir, "concat_list.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                abs_seg = os.path.abspath(seg).replace("\\", "/")
                f.write(f"file '{abs_seg}'\n")

        master_out = os.path.join(output_dir, "master_render.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            master_out
        ], check=True, capture_output=True)

        normalized_out = master_out.replace("\\", "/")
        storyboard_data["status"] = "completed"
        storyboard_data["video_path"] = normalized_out
        save_stage_checkpoint(self.job_id, "storyboard", storyboard_data)
        update_job(self.job_id, status="completed", progress=1.0, step="Master Video Ready", result=normalized_out)
        self._log(f"Master video rendered successfully: {normalized_out}")
        manager.broadcast(self.job_id, {
            "type": "render_complete",
            "video_path": normalized_out,
            "data": storyboard_data
        })
        return normalized_out

    @classmethod
    def regenerate_single_scene(cls, job_id: str, scene_number: int, instruction: str) -> Dict[str, Any]:
        """Regenerates a single scene in the storyboard and persists the updated plan."""
        from server.core.agent_system import ScriptScene, CreativeDirectorAgent

        storyboard_data = get_stage_checkpoint(job_id, "storyboard")
        if not storyboard_data:
            job = get_job(job_id)
            if job and job.get("result"):
                try:
                    res = json.loads(job["result"])
                    storyboard_data = res
                except Exception:
                    pass
        if not storyboard_data or "scenes" not in storyboard_data:
            raise ValueError(f"No storyboard plan found for job {job_id}")

        director = CreativeDirectorAgent()
        scenes = [ScriptScene(**s) for s in storyboard_data["scenes"]]
        target_scene_idx = next((i for i, s in enumerate(scenes) if s.scene_number == scene_number), None)
        if target_scene_idx is None:
            raise ValueError(f"Scene number {scene_number} not found in storyboard")

        # Regenerate target scene
        updated_scene = director.revise_scene(scenes[target_scene_idx], instruction)
        scenes[target_scene_idx] = updated_scene

        storyboard_data["scenes"] = [s.model_dump() for s in scenes]
        save_stage_checkpoint(job_id, "storyboard", storyboard_data)
        update_job(job_id, result=json.dumps(storyboard_data))
        manager.broadcast(job_id, {"type": "storyboard_updated", "data": storyboard_data})
        return storyboard_data

    @classmethod
    def update_storyboard_scenes(cls, job_id: str, updated_scenes: list) -> Dict[str, Any]:
        """Updates storyboard scene modifications (locks, edits) and persists them."""
        storyboard_data = get_stage_checkpoint(job_id, "storyboard")
        if not storyboard_data:
            job = get_job(job_id)
            if job and job.get("result"):
                try:
                    storyboard_data = json.loads(job["result"])
                except Exception:
                    pass
        if not storyboard_data:
            storyboard_data = {"scenes": []}

        storyboard_data["scenes"] = updated_scenes
        save_stage_checkpoint(job_id, "storyboard", storyboard_data)
        update_job(job_id, result=json.dumps(storyboard_data))
        manager.broadcast(job_id, {"type": "storyboard_updated", "data": storyboard_data})
        return storyboard_data
