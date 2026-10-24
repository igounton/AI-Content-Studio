import os
import re
import json
import logging
import subprocess
import pysubs2
import cv2
import numpy as np
import requests
from server.core.youtube_downloader import download_youtube_video
from pipeline import generate_captions
from api_clients import GoogleClient
from config import load_config

def fetch_b_roll_video(query: str, api_key: str, output_path: str) -> bool:
    if not api_key: return False
    try:
        url = f"https://pixabay.com/api/videos/?key={api_key}&q={query}&video_type=film&per_page=3"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("hits") and len(data["hits"]) > 0:
            video_url = data["hits"][0]["videos"]["medium"]["url"]
            video_data = requests.get(video_url, timeout=20).content
            with open(output_path, "wb") as f:
                f.write(video_data)
            return True
    except Exception as e:
        logging.error(f"Failed to fetch B-Roll: {e}")
    return False

def compute_audio_energy_peaks(video_path: str, sample_interval_sec: float = 1.0) -> list:
    """Computes audio RMS energy envelope per interval to detect excitement, vocal spikes, and volume peaks."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(video_path)
        chunk_len_ms = int(sample_interval_sec * 1000)
        energies = []
        for i in range(0, len(audio), chunk_len_ms):
            chunk = audio[i:i + chunk_len_ms]
            energies.append(chunk.rms)
        if not energies:
            return []
        max_rms = max(energies) or 1
        return [float(e) / float(max_rms) for e in energies]
    except Exception as e:
        logging.warning(f"Audio energy profile note: {e}")
        return []

def compute_visual_motion_profile(video_path: str, sample_interval_sec: float = 1.0) -> list:
    """Computes visual scene change & motion intensity envelope using OpenCV HSV histogram differences."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps * sample_interval_sec))
        
        scores = []
        prev_hist = None
        frame_idx = 0
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                scores.append(max(0.0, float(diff)))
            else:
                scores.append(0.0)
                
            prev_hist = hist
            frame_idx += frame_interval
            
        cap.release()
        if not scores:
            return []
        max_s = max(scores) or 1
        return [float(s) / float(max_s) for s in scores]
    except Exception as e:
        logging.warning(f"Visual motion profile note: {e}")
        return []

def get_multimodal_boost(start: float, end: float, audio_profile: list, visual_profile: list, interval: float = 1.0) -> float:
    """Calculates multimodal excitement boost (0 to 15 bonus score) combining audio volume peaks and visual cuts."""
    if not audio_profile and not visual_profile:
        return 0.0
    s_idx = int(start / interval)
    e_idx = max(s_idx + 1, int(end / interval))
    
    a_score = 0.0
    if audio_profile and s_idx < len(audio_profile):
        clip_audio = audio_profile[s_idx:min(e_idx, len(audio_profile))]
        if clip_audio:
            a_score = sum(clip_audio) / len(clip_audio)
            
    v_score = 0.0
    if visual_profile and s_idx < len(visual_profile):
        clip_vis = visual_profile[s_idx:min(e_idx, len(visual_profile))]
        if clip_vis:
            v_score = sum(clip_vis) / len(clip_vis)
            
    return (a_score * 0.65 + v_score * 0.35) * 15.0

def render_scene_aware_clip(video_path: str, start_time: float, end_time: float, out_path: str):
    """OpusClip-style AI Active Speaker Tracking using Scene-Aware Cropping."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        segment_path = tmp.name
        
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time),
        "-i", video_path, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", segment_path
    ], check=True, capture_output=True)
    
    cap = cv2.VideoCapture(segment_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = int(height * 9 / 16)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cuts = []
    current_cut_start = 0.0
    current_crop_x = None
    
    frame_idx = 0
    skip_frames = int(fps / 2) # Sample twice per second
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            crop_x = max(0, min((x + w // 2) - target_w // 2, width - target_w))
            
            if current_crop_x is None:
                current_crop_x = crop_x
            elif abs(crop_x - current_crop_x) > (width * 0.18):
                cut_time = frame_idx / fps
                # Apply cut hysteresis: minimum 1.5 seconds between shot cuts to prevent jitter
                if (cut_time - current_cut_start) >= 1.5:
                    cuts.append({"start": current_cut_start, "end": cut_time, "crop_x": current_crop_x})
                    current_cut_start = cut_time
                    current_crop_x = crop_x
                    
        frame_idx += skip_frames
        
    cap.release()
    
    duration = end_time - start_time
    if current_crop_x is None:
        current_crop_x = (width - target_w) // 2
    cuts.append({"start": current_cut_start, "end": duration, "crop_x": current_crop_x})
    
    if len(cuts) == 1:
        subprocess.run([
            "ffmpeg", "-y", "-i", segment_path,
            "-vf", f"crop={target_w}:{height}:{int(cuts[0]['crop_x'])}:0,scale=720:1280",
            "-c:v", "libx264", "-c:a", "copy", out_path
        ], check=True, capture_output=True)
    else:
        filter_parts = []
        concat_inputs = ""
        for i, cut in enumerate(cuts):
            filter_parts.append(f"[0:v]trim=start={cut['start']}:end={cut['end']},setpts=PTS-STARTPTS,crop={target_w}:{height}:{int(cut['crop_x'])}:0,scale=720:1280[v{i}]")
            concat_inputs += f"[v{i}]"
            
        filter_parts.append(f"{concat_inputs}concat=n={len(cuts)}:v=1:a=0[outv]")
        subprocess.run([
            "ffmpeg", "-y", "-i", segment_path,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[outv]", "-map", "0:a",
            "-c:v", "libx264", "-c:a", "copy", out_path
        ], check=True, capture_output=True)
        
    os.remove(segment_path)

def extract_clip_transcript_segments(word_timestamps: list, clip_start: float, clip_end: float) -> list:
    """Slices word timestamps for [clip_start, clip_end] into sentence/phrase segments for TimelineEditor."""
    clip_words = [
        w for w in word_timestamps
        if float(w.get("end", 0)) > clip_start and float(w.get("start", 0)) < clip_end
    ]
    if not clip_words:
        return []
    segments = []
    cur_words = []
    seg_start = max(0.0, round(float(clip_words[0]["start"]) - clip_start, 2))
    seg_idx = 1
    for i, w in enumerate(clip_words):
        word_text = str(w.get("word", "")).strip()
        if not word_text:
            continue
        if not cur_words:
            seg_start = max(0.0, round(float(w["start"]) - clip_start, 2))
        cur_words.append(word_text)
        seg_end = max(seg_start + 0.3, round(float(w["end"]) - clip_start, 2))
        ends_sentence = word_text.endswith((".", "!", "?"))
        if len(cur_words) >= 8 or ends_sentence or (seg_end - seg_start >= 4.5) or i == len(clip_words) - 1:
            segments.append({
                "id": f"seg_{seg_idx}",
                "start": seg_start,
                "end": seg_end,
                "text": " ".join(cur_words),
                "deleted": False
            })
            seg_idx += 1
            cur_words = []
    return segments

def analyze_video(source: str, is_local: bool, job_id: str, update_job_callback, num_clips: int = 10):
    try:
        config = load_config()
        client = GoogleClient(config)
        output_dir = os.path.join("workspace", f"clipper_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        if is_local:
            update_job_callback(job_id, step="Processing Local Video", progress=0.1)
            video_path = source
        else:
            update_job_callback(job_id, step="Downloading Video", progress=0.1)
            video_path = download_youtube_video(source, output_dir)
        
        update_job_callback(job_id, step="Analyzing Audio Peaks & Scene Cuts", progress=0.25)
        audio_profile = compute_audio_energy_peaks(video_path)
        visual_profile = compute_visual_motion_profile(video_path)

        update_job_callback(job_id, step="Transcribing Video", progress=0.35)
        word_timestamps = generate_captions(video_path, None, "English")
        
        if not word_timestamps:
            raise Exception("Transcription failed or returned no words.")
            
        transcript_chunks = []
        current_chunk_words = []
        chunk_start = 0.0
        
        for w in word_timestamps:
            if not current_chunk_words:
                chunk_start = w['start']
            current_chunk_words.append(w['word'])
            
            if w['end'] - chunk_start >= 10.0:
                transcript_chunks.append(f"[{chunk_start:.1f}s - {w['end']:.1f}s]: {' '.join(current_chunk_words)}")
                current_chunk_words = []
                
        if current_chunk_words:
            transcript_chunks.append(f"[{chunk_start:.1f}s - {word_timestamps[-1]['end']:.1f}s]: {' '.join(current_chunk_words)}")
            
        full_transcript = "\n".join(transcript_chunks)
        
        update_job_callback(job_id, step="AI Finding Viral Clips with Multimodal Fusion", progress=0.5)
        
        # Hierarchical whole-video chunk analysis (preserves late video revelations)
        window_size = 12000
        step_size = 9000
        transcript_windows = []
        if len(full_transcript) <= window_size:
            transcript_windows.append(full_transcript)
        else:
            for offset in range(0, len(full_transcript), step_size):
                window = full_transcript[offset:offset + window_size]
                if window.strip():
                    transcript_windows.append(window)

        all_candidate_clips = []
        for w_idx, win in enumerate(transcript_windows):
            prompt = f"""You are an elite, highly intelligent Social Media Marketer and Content Curator.
Your objective is to analyze the following video transcript segment and extract high-retention short clips (strictly between 15 and 60 seconds).

CRITICAL CRITERIA FOR CLIPS:
1. The Hook: Immediate, scroll-stopping opening statement.
2. The Value: Builds curiosity, reveals surprising insights, or provides strong emotional payoff.
3. Narrative Completeness: Understandable as a standalone video without watching the full source.

Return your response ONLY as a JSON list of objects, with no markdown formatting.
Each object must match this schema:
{{
  "start": float (seconds),
  "end": float (seconds),
  "title": "Clickbaity short title",
  "score": int (1-100 overall virality),
  "hook_score": int (1-100),
  "retention_score": int (1-100),
  "reason": "1 sentence why it's viral",
  "seo_title": "Highly optimized YouTube Shorts Title #shorts",
  "seo_description": "Engaging description...",
  "seo_tags": "shorts, viral, podcast",
  "broll": [
    {{"start_offset": int (seconds from clip start), "end_offset": int, "subject": "visual search term (e.g. technology, money)"}}
  ]
}}
Do not hallucinate timestamps.

Transcript window:
{win}
"""
            try:
                response = client._generate_text(prompt).strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                parsed = json.loads(response.strip())
                if isinstance(parsed, list):
                    all_candidate_clips.extend(parsed)
            except Exception as e:
                logging.warning(f"Window {w_idx} parsing notice: {e}")

        # Deduplicate and sort clips by score with multimodal boost
        seen_ranges = set()
        unique_clips = []
        for c in all_candidate_clips:
            try:
                st = round(float(c.get("start", 0)), 1)
                en = round(float(c.get("end", 0)), 1)
                is_dup = any(abs(st - ex_st) < 8 for ex_st, _ in seen_ranges)
                if not is_dup and en > st:
                    seen_ranges.add((st, en))
                    
                    # Apply multimodal boost from audio volume peaks and visual cuts
                    boost = get_multimodal_boost(st, en, audio_profile, visual_profile)
                    c["multimodal_boost"] = round(boost, 1)
                    c["score"] = min(99, int(c.get("score", 70) + boost))
                    if boost > 6.0:
                        c["reason"] = f"{c.get('reason', '')} [Multimodal verified: Audio peak + Cut dynamics]"
                        
                    c["transcript_segments"] = extract_clip_transcript_segments(word_timestamps, st, en)
                    unique_clips.append(c)
            except Exception:
                continue

        unique_clips.sort(key=lambda x: int(x.get("score", 50)), reverse=True)
        clips_data = unique_clips[:num_clips]
        
        if not clips_data:
            # Fallback segment if API was unavailable
            clips_data = [{
                "start": 0.0,
                "end": min(30.0, float(word_timestamps[-1]['end'])),
                "title": "Highlight Clip",
                "score": 85,
                "hook_score": 85,
                "retention_score": 80,
                "reason": "High interest opening hook",
                "seo_title": "Must Watch Moment #shorts",
                "seo_description": "Key highlight from the full episode.",
                "seo_tags": "viral, shorts",
                "broll": [],
                "transcript_segments": extract_clip_transcript_segments(word_timestamps, 0.0, min(30.0, float(word_timestamps[-1]['end'])))
            }]
            
        update_job_callback(job_id, step="Awaiting Selection", progress=0.6, result=json.dumps({"type": "selection", "clips": clips_data, "video_path": video_path}))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def render_youtube_clips(job_id: str, video_path: str, selected_clips: list, update_job_callback):
    try:
        config = load_config()
        client = GoogleClient(config)
        output_dir = os.path.join("workspace", f"clipper_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        update_job_callback(job_id, step="Cropping & Captioning", progress=0.7)
        
        generated_clips = []
        total_clips = len(selected_clips)
        
        for idx, clip in enumerate(selected_clips):
            try:
                start = float(clip.get("start", 0))
                end = float(clip.get("end", 30))
                title = str(clip.get("title") or f"Clip {idx+1}")
                
                safe_title = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(" ", "_"))
                clip_name = f"{safe_title}.mp4"
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping malformed clip data: {clip} - {e}")
                continue
                
            out_path = os.path.join(output_dir, clip_name)
            
            # Use OpusClip-style AI Active Speaker Tracking
            render_scene_aware_clip(video_path, start, end, out_path)

            # Apply transcript ripple cuts if user deleted segments in TimelineEditor
            kept_segments = clip.get("kept_segments")
            if isinstance(kept_segments, list) and len(kept_segments) > 0:
                try:
                    ripple_path = os.path.join(output_dir, f"ripple_{clip_name}")
                    v_parts = []
                    a_parts = []
                    concat_pairs = ""
                    for s_i, seg in enumerate(kept_segments):
                        s_st = max(0.0, float(seg.get("start", 0.0)))
                        s_en = max(s_st + 0.2, float(seg.get("end", s_st + 1.0)))
                        v_parts.append(f"[0:v]trim=start={s_st}:end={s_en},setpts=PTS-STARTPTS[rv{s_i}]")
                        a_parts.append(f"[0:a]atrim=start={s_st}:end={s_en},asetpts=PTS-STARTPTS[ra{s_i}]")
                        concat_pairs += f"[rv{s_i}][ra{s_i}]"
                    fc = ";".join(v_parts + a_parts + [f"{concat_pairs}concat=n={len(kept_segments)}:v=1:a=1[outv][outa]"])
                    subprocess.run([
                        "ffmpeg", "-y", "-i", out_path,
                        "-filter_complex", fc,
                        "-map", "[outv]", "-map", "[outa]",
                        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
                        ripple_path
                    ], check=True, capture_output=True)
                    if os.path.exists(ripple_path) and os.path.getsize(ripple_path) > 0:
                        out_path = ripple_path
                except Exception as ripple_err:
                    logging.warning(f"Transcript ripple splice skipped for {clip_name}: {ripple_err}")
            
            # --- PHASE 4: DYNAMIC B-ROLL TIMELINE ---
            try:
                pixabay_key = config.get("PIXABAY_API_KEY", "")
                broll_timeline = clip.get("broll", [])
                
                if pixabay_key and broll_timeline:
                    current_vid = out_path
                    for b_idx, b_item in enumerate(broll_timeline):
                        b_subj = str(b_item.get("subject", "")).lower()
                        b_start = b_item.get("start_offset", 2)
                        b_end = b_item.get("end_offset", 5)
                        
                        if not b_subj or b_subj == "none": continue
                        
                        b_roll_path = os.path.join(output_dir, f"broll_{idx}_{b_idx}.mp4")
                        if fetch_b_roll_video(b_subj, pixabay_key, b_roll_path):
                            b_roll_overlay_path = os.path.join(output_dir, f"broll_overlay_{idx}_{b_idx}.mp4")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", current_vid, "-i", b_roll_path,
                                "-filter_complex", 
                                f"[1:v]crop=ih*9/16:ih,scale=720:1280[broll];[0:v][broll]overlay=enable='between(t,{b_start},{b_end})'",
                                "-c:a", "copy",
                                b_roll_overlay_path
                            ], check=True, capture_output=True)
                            current_vid = b_roll_overlay_path
                    out_path = current_vid
            except Exception as e:
                logging.error(f"Dynamic B-Roll injection failed for {clip_name}: {e}")
                
            # --- PHASE 4.5: AUDIO DUCKING (BGM) ---
            try:
                bgm_path = "workspace/bgm.mp3"
                if os.path.exists(bgm_path):
                    bgm_out_path = os.path.join(output_dir, f"bgm_{clip_name}")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", out_path, "-i", bgm_path,
                        "-filter_complex",
                        "[1:a]volume=0.08[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2",
                        "-c:v", "copy",
                        bgm_out_path
                    ], check=True, capture_output=True)
                    out_path = bgm_out_path
            except Exception as e:
                logging.error(f"Audio Ducking failed for {clip_name}: {e}")
            
            theme = config.get("CAPTION_THEME", "default")
            font = config.get("CAPTION_FONT", "Arial")
            font_size = int(config.get("CAPTION_FONT_SIZE", 22))
            
            if theme == "viral_yellow":
                primarycolor = pysubs2.Color(255, 255, 0, 255)
                backcolor = pysubs2.Color(0, 0, 0, 255)
                outline = 2.0
            elif theme == "neon_cyber":
                primarycolor = pysubs2.Color(0, 255, 255, 255)
                backcolor = pysubs2.Color(255, 0, 255, 100)
                outline = 1.0
            elif theme == "black_white":
                primarycolor = pysubs2.Color(0, 0, 0, 255)
                backcolor = pysubs2.Color(255, 255, 255, 255)
                outline = 2.0
            else:
                primarycolor = pysubs2.Color(255, 255, 255, 255)
                backcolor = pysubs2.Color(0, 0, 0, 150)
                outline = 1.5

            ass_path = os.path.join(output_dir, f"{safe_title}.ass")
            generate_captions(out_path, ass_path, "English", style_opts={
                "fontname": font, 
                "fontsize": font_size, 
                "alignment": 2, 
                "marginv": 60,
                "bold": True,
                "outline": outline,
                "shadow": 1,
                "primarycolor": primarycolor,
                "backcolor": backcolor
            })
            
            final_out_path = os.path.join(output_dir, f"Final_{clip_name}")
            subs_path = os.path.abspath(ass_path).replace('\\', '/').replace(':', '\\:')
            
            subprocess.run([
                "ffmpeg", "-y", "-i", out_path,
                "-vf", f"subtitles='{subs_path}'",
                "-c:v", "libx264", "-c:a", "copy",
                final_out_path
            ], check=True, capture_output=True)
            
            generated_clips.append({
                "title": title,
                "path": final_out_path.replace("\\", "/"),
                "seo_title": clip.get("seo_title", title),
                "seo_description": clip.get("seo_description", ""),
                "seo_tags": clip.get("seo_tags", ""),
            })
            
            progress = 0.7 + (0.3 * ((idx + 1) / total_clips))
            update_job_callback(job_id, step=f"Rendered {idx+1}/{total_clips} Clips", progress=progress)
            
        update_job_callback(job_id, step="Finished", progress=1.0, result=json.dumps({"type": "finished", "clips": generated_clips}))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Clipper Error: {e}", exc_info=True)
        update_job_callback(job_id, step="Failed", progress=1.0, error=str(e))
