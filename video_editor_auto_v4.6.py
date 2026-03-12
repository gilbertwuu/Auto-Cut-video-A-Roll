#!/usr/bin/env python3
"""
Video Auto Editor v4.7
Copyright (C) 2026

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY. See the LICENSE file for details.

Scenario A: Single video (silence detection -> segment identification -> scoring
            -> transcription -> fluency analysis -> dedup -> clip)
Scenario B: Batch processing -> cross-video dedup -> concatenation

Dependencies: FFmpeg, openai-whisper
"""
import subprocess, re, os, sys, json, glob, difflib, datetime, shutil
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- Config ---
CONFIG = {
    "silence_noise": -30,           # dB, lower = stricter
    "silence_duration": 0.8,        # seconds, min silence length
    "min_score": 90,                # min base score (max 100)
    "min_duration": 15,             # min segment duration (seconds)
    "buffer_start": 1,              # buffer before start (seconds)
    "buffer_end": 3,                # buffer after end (seconds)
    "crf": 18,                      # video quality (18=visually lossless)
    "preset": "fast",               # encoding speed
    "audio_bitrate": "192k",        # audio bitrate
    "penalty_repeat": 5,            # penalty per repeat
    "penalty_stutter": 3,            # penalty per stutter
    "penalty_interrupt": 10,        # sudden interruption penalty
    "bonus_natural_end": 5,         # natural ending bonus
    "bonus_completeness_max": 3,    # completeness bonus cap
    "duplicate_threshold": 0.7,     # content similarity threshold
}

# --- Data structures ---
@dataclass
class Segment:
    """Video segment (non-silent interval)"""
    index: int
    start_time: float
    end_time: float
    duration: float
    score_start: float = 0
    score_end: float = 0
    score_fluency: float = 0
    score_rhythm: float = 0
    total_score: float = 0
    internal_silences: List[Tuple[float, float]] = field(default_factory=list)
    interruption_count: int = 0
    interruption_duration: float = 0
    transcript: str = ""
    repeat_count: int = 0
    stutter_count: int = 0
    is_natural_end: bool = False
    is_interrupted: bool = False
    adjusted_score: float = 0
    is_duplicate: bool = False
    duplicate_with: List[int] = field(default_factory=list)

@dataclass
class ClipInfo:
    """Single video rough-cut result, for cross-video dedup"""
    video_name: str
    clip_path: str
    transcript: str
    adjusted_score: float
    is_natural_end: bool
    duration: float
    is_cross_duplicate: bool = False
    duplicate_of: str = ""

# --- Module 1: Silence detection ---
def detect_silence(video_path):
    """Detect silence spans using FFmpeg silencedetect filter"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={CONFIG['silence_noise']}dB:d={CONFIG['silence_duration']}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    starts = re.findall(r'silence_start: ([\d.]+)', result.stderr)
    ends = re.findall(r'silence_end: ([\d.]+)', result.stderr)
    return [(float(starts[i]), float(ends[i])) for i in range(min(len(starts), len(ends)))]

# --- Module 2: Segment identification ---
def identify_segments(silences, total_duration):
    """Split video into segments by silence (>=1s non-silent intervals)"""
    if not silences:
        return [Segment(index=0, start_time=0, end_time=total_duration, duration=total_duration)]

    segments = []
    idx = 0

    first_end = silences[0][0]
    if first_end > 1.0:
        segments.append(Segment(index=idx, start_time=0, end_time=first_end, duration=first_end))
        idx += 1

    for i in range(len(silences) - 1):
        start, end = silences[i][1], silences[i+1][0]
        duration = end - start
        if duration >= 1.0:
            segments.append(Segment(index=idx, start_time=start, end_time=end, duration=duration))
            idx += 1

    last_start = silences[-1][1]
    if total_duration - last_start > 1.0:
        segments.append(Segment(index=idx, start_time=last_start, end_time=total_duration,
                                duration=total_duration - last_start))

    return segments

# --- Module 3: Scoring (4 dims x 25 pts = 100) ---
def _score_boundary(silences, time_point, total_duration, is_start):
    """Score segment boundary clarity (start or end), return 0-25"""
    if is_start:
        nearby = [s for s in silences if abs(s[1] - time_point) < 0.1]
    else:
        nearby = [s for s in silences if abs(s[0] - time_point) < 0.1]

    if nearby:
        dur = nearby[0][1] - nearby[0][0]
        if dur >= 1.0: return 25
        if dur >= 0.5: return 20
        return 10

    if is_start and time_point < 0.5:
        return 15
    if not is_start and abs(time_point - total_duration) < 0.5:
        return 15
    return 5

def score_segment(seg, silences, total_duration):
    """4-dimension scoring: clear start/end + mid fluency + natural rhythm"""
    seg.score_start = _score_boundary(silences, seg.start_time, total_duration, is_start=True)
    seg.score_end = _score_boundary(silences, seg.end_time, total_duration, is_start=False)

    # Mid fluency (25 pts)
    internal = [s for s in silences
                if s[0] > seg.start_time + 0.1 and s[1] < seg.end_time - 0.1]
    seg.internal_silences = internal
    seg.interruption_count = len(internal)
    seg.interruption_duration = sum(e - s for s, e in internal)

    if seg.interruption_count == 0:      seg.score_fluency = 25
    elif seg.interruption_count <= 2:    seg.score_fluency = 20
    elif seg.interruption_count <= 4:    seg.score_fluency = 15
    else: seg.score_fluency = max(5, 25 - seg.interruption_count * 3)

    # Natural rhythm (25 pts): pause ratio (15) + max single pause (10) + short-segment cap
    score_rhythm = 0
    if seg.duration > 0:
        ratio = seg.interruption_duration / seg.duration
        score_rhythm += 15 if ratio < 0.05 else 12 if ratio < 0.10 else 8 if ratio < 0.20 else 4

        max_pause = max((e - s for s, e in internal), default=0)
        score_rhythm += 10 if max_pause < 0.8 else 7 if max_pause < 1.5 else 4 if max_pause < 2.5 else 0

        if seg.duration < 8:       score_rhythm = min(score_rhythm, 15)
        elif seg.duration < 15:    score_rhythm = min(score_rhythm, 20)

    seg.score_rhythm = score_rhythm
    seg.total_score = seg.score_start + seg.score_end + seg.score_fluency + seg.score_rhythm
    return seg

# --- Module 4: Transcription (Whisper CLI) ---
def transcribe_segment(video_path, seg, work_dir):
    """Extract segment audio and transcribe via Whisper"""
    audio_path = os.path.join(work_dir, f"segment_{seg.index}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(seg.start_time), "-to", str(seg.end_time),
        "-vn", "-ar", "16000", "-ac", "1", audio_path
    ], capture_output=True, text=True)

    if not os.path.exists(audio_path):
        print(f"    ⚠️  Audio extraction failed: segment_{seg.index}")
        return ""
    try:
        subprocess.run([
            "python3", "-m", "whisper", audio_path,
            "--model", "small", "--language", "zh",
            "--output_format", "txt", "--output_dir", work_dir
        ], capture_output=True, text=True, timeout=120)
        txt_path = audio_path.replace('.wav', '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        print(f"    ⚠️  Transcript file not generated: segment_{seg.index}")
        return ""
    except Exception as e:
        print(f"    ⚠️  Transcription failed: segment_{seg.index}: {e}")
        return ""

# --- Module 5: Fluency analysis ---
def analyze_fluency(transcript):
    """Analyze transcript, return (repeat_count, stutter_count, is_natural_end, is_interrupted)"""
    if not transcript:
        return 0, 0, False, False

    text = re.sub(r'(?i)\bwhisper\b', '', transcript.strip()).strip()

    # Sliding-window phrase repeat detection (2-4 char phrases in 10-char window)
    text_clean = re.sub(r'[^\w]', '', text)
    repeat_count, i, window = 0, 0, 10
    while i < len(text_clean) - 2:
        found = False
        for length in [4, 3, 2]:
            if i + length > len(text_clean):
                continue
            chunk = text_clean[i:i+length]
            area = text_clean[i+length : i+length+window]
            if chunk in area:
                repeat_count += 1
                i += length + area.index(chunk) + length
                found = True
                break
        if not found:
            i += 1

    # Stutter detection (patterns for Chinese: 嗯啊呃=um/uh, 那个=that, 就是说=I mean, ellipsis)
    stutter_count = sum(len(re.findall(p, text)) for p in [r'[嗯啊呃]', r'那个', r'就是说', r'\.{2,}', r'…'])

    # Sudden interruption (Chinese connective/incomplete markers)
    interrupt_re = r'(的时候|然后|但是|如果|因为|而且|所以|就是|其实|那么|或者|并且|还是|不过|包括|比如说|另外|接下来|还有就是|就是说)$'
    is_interrupted = bool(re.search(interrupt_re, text))

    # Natural end detection
    has_punctuation = bool(re.search(r'[。！？]$', text))
    is_connective_end = bool(re.search(interrupt_re, text))
    # Natural end patterns (Chinese: questions, summaries, farewells)
    special_natural_patterns = [
        r'怎么[^。！？]*[呢？]$', r'什么[^。！？]*[呢？]$', r'为什么[^。！？]*[呢？]$',
        r'就是这样[。！？]*$', r'其实有很多[的。]*$',
        r'拜拜[^\w]*$', r'再见[^\w]*$', r'今天就到这[^\w]*$',
        r'分享给大家[^\w]*$', r'希望对你[也]*有帮助[^\w]*$',
    ]
    is_natural_end = (has_punctuation and not is_connective_end) or any(re.search(p, text) for p in special_natural_patterns)
    if is_interrupted:
        is_natural_end = False

    return repeat_count, stutter_count, is_natural_end, is_interrupted

# --- Module 6: Adjusted score ---
def calculate_adjusted_score(seg):
    """Apply fluency penalties/bonuses to base score, normalize to 0-100"""
    adjusted = seg.total_score
    duration_factor = max(1.0, seg.duration / 30.0)
    adjusted -= (seg.repeat_count / duration_factor) * CONFIG['penalty_repeat']
    adjusted -= (seg.stutter_count / duration_factor) * CONFIG['penalty_stutter']
    if seg.is_interrupted:
        adjusted -= CONFIG['penalty_interrupt']
    if seg.is_natural_end:
        adjusted += CONFIG['bonus_natural_end']
    if seg.is_natural_end and not seg.is_interrupted:
        adjusted += max(0, CONFIG['bonus_completeness_max'] * (1 - abs(seg.duration - 60) / 60))
    return max(0, min(100, adjusted))

# --- Module 7: Content dedup (generic grouping) ---
def _find_duplicate_groups(items, get_text):
    """Group items by text similarity (pairwise comparison)"""
    groups = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            t1, t2 = get_text(items[i]), get_text(items[j])
            if not t1 or not t2:
                continue
            if difflib.SequenceMatcher(None, t1, t2).ratio() > CONFIG['duplicate_threshold']:
                merged = False
                for group in groups:
                    if i in group or j in group:
                        group.update([i, j])
                        merged = True
                        break
                if not merged:
                    groups.append({i, j})
    return groups

def check_duplicate_content(candidates):
    """Within-video dedup: group candidates, keep best per group, mark rest as duplicate"""
    groups = _find_duplicate_groups(candidates, lambda s: s.transcript)
    for group in groups:
        for i, j in [(i, j) for i in group for j in group if i < j]:
            print(f"    ⚠️  segment_{candidates[i].index} and segment_{candidates[j].index} content similar")
        best = max(group, key=lambda idx: (candidates[idx].is_natural_end, candidates[idx].adjusted_score, candidates[idx].index))
        for idx in group:
            if idx != best:
                candidates[idx].is_duplicate = True
                candidates[idx].duplicate_with.append(candidates[best].index)
    return candidates

def cross_video_dedup(clips):
    """Cross-video dedup: compare clip transcripts, mark duplicates"""
    if len(clips) < 2:
        return clips
    groups = _find_duplicate_groups(clips, lambda c: c.transcript)
    for group in groups:
        for i, j in [(i, j) for i in group for j in group if i < j]:
            print(f"    ⚠️  {clips[i].video_name} and {clips[j].video_name} content similar")
        best = max(group, key=lambda idx: (clips[idx].is_natural_end, clips[idx].adjusted_score, clips[idx].video_name))
        for idx in group:
            if idx != best:
                clips[idx].is_cross_duplicate = True
                clips[idx].duplicate_of = clips[best].video_name
    return clips

# --- Module 8: Layered selection ---
def select_best_segment(candidates):
    """Layered selection: natural end -> fluency -> adjusted score -> duration/index"""
    if not candidates:
        return None

    pool = [c for c in candidates if not c.is_duplicate] or list(candidates)
    all_unnatural = not any(s.is_natural_end for s in pool)

    natural_end = [s for s in pool if s.is_natural_end]
    if natural_end:
        pool = natural_end

    pool.sort(key=lambda s: (s.stutter_count + s.repeat_count) / max(1.0, s.duration / 30.0))
    best_rate = (pool[0].stutter_count + pool[0].repeat_count) / max(1.0, pool[0].duration / 30.0)
    pool = [s for s in pool if ((s.stutter_count + s.repeat_count) / max(1.0, s.duration / 30.0)) - best_rate <= 1.5]

    pool.sort(key=lambda s: s.adjusted_score, reverse=True)
    pool = [s for s in pool if s.adjusted_score == pool[0].adjusted_score]

    if len(pool) > 1:
        if all_unnatural:
            pool.sort(key=lambda s: s.index, reverse=True)
        else:
            pool.sort(key=lambda s: s.duration, reverse=True)

    return pool[0]

# --- Module 9: FFmpeg operations ---
def clip_segment(video_path, seg, output_path):
    """Clip target segment with FFmpeg (with start/end buffer)"""
    start = max(0, seg.start_time - CONFIG['buffer_start'])
    end = seg.end_time + CONFIG['buffer_end']
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start), "-to", str(end),
        "-c:v", "libx264", "-crf", str(CONFIG['crf']), "-preset", CONFIG['preset'],
        "-c:a", "aac", "-b:a", CONFIG['audio_bitrate'],
        output_path
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0

def concat_videos(clip_paths, output_path):
    """Concatenate video list using FFmpeg concat protocol"""
    list_file = output_path + ".list.txt"
    with open(list_file, 'w') as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
        "-c:v", "libx264", "-crf", str(CONFIG['crf']), "-preset", CONFIG['preset'],
        "-c:a", "aac", "-b:a", CONFIG['audio_bitrate'],
        output_path
    ]
    ok = subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    if os.path.exists(list_file):
        os.remove(list_file)
    return ok

# --- Scenario A: Single video ---
def process_single_video(video_path, output_dir, work_dir, batch_mode=False):
    """
    Process single video, return ClipInfo; None on failure.
    batch_mode=True skips individual report (Scenario B generates one).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_work = os.path.join(work_dir, video_name)
    os.makedirs(video_work, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Processing: {video_name}" if batch_mode else f"  Video Auto Editor v4.7 - Scenario A\n  Input: {video_path}")
    print(f"{'='*60}\n")

    # Step 1: Get video info
    print("📋 Step 1: Getting video info...")
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        total_duration = float(json.loads(result.stdout)['format']['duration'])
    except Exception as e:
        print(f"   ❌ Failed to get video info: {e}")
        return None
    print(f"   Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Step 2: Silence detection
    print("\n🔇 Step 2: Silence detection...")
    silences = detect_silence(video_path)
    print(f"   Detected {len(silences)} silence spans")

    # Step 3: Segment identification
    print("\n📝 Step 3: Segment identification...")
    segments = identify_segments(silences, total_duration)
    print(f"   Identified {len(segments)} segments")

    # Step 4: Scoring
    print("\n⭐ Step 4: Scoring...")
    for seg in segments:
        score_segment(seg, silences, total_duration)
        print(f"   segment_{seg.index}: {seg.start_time:.1f}s-{seg.end_time:.1f}s "
              f"({seg.duration:.1f}s) score={seg.total_score}")

    # Step 5: Filter candidates
    print(f"\n🔍 Step 5: Filtering candidates (min_score={CONFIG['min_score']}, min_duration={CONFIG['min_duration']}s)...")
    candidates = [s for s in segments if s.total_score >= CONFIG['min_score'] and s.duration >= CONFIG['min_duration']]
    print(f"   {len(candidates)} candidate segments")

    if not candidates:
        print("\n⚠️  No candidates meet criteria, lowering standards...")
        candidates = sorted(segments, key=lambda s: s.total_score, reverse=True)[:5]
        print(f"   Selected top {len(candidates)} segments by score")

    # Step 6: Transcription
    print("\n🎤 Step 6: Transcribing candidates...")
    whisper_available = True
    try:
        if subprocess.run(["python3", "-m", "whisper", "--help"], capture_output=True, text=True, timeout=10).returncode != 0:
            whisper_available = False
    except Exception:
        whisper_available = False

    if whisper_available:
        for seg in candidates:
            print(f"   Transcribing segment_{seg.index}...")
            seg.transcript = transcribe_segment(video_path, seg, video_work)
            if seg.transcript:
                preview = seg.transcript[:50] + "..." if len(seg.transcript) > 50 else seg.transcript
                print(f"   ✅ [{preview}]")
    else:
        print("   ⚠️  Whisper not installed, skipping transcription, using audio-only scoring")

    # Step 7: Fluency analysis
    print("\n📊 Step 7: Fluency analysis...")
    for seg in candidates:
        if seg.transcript:
            seg.repeat_count, seg.stutter_count, seg.is_natural_end, seg.is_interrupted = analyze_fluency(seg.transcript)
        seg.adjusted_score = calculate_adjusted_score(seg)
        status = (" ✅natural end" if seg.is_natural_end else "") + (" ❌interrupted" if seg.is_interrupted else "")
        print(f"   segment_{seg.index}: base={seg.total_score} adjusted={seg.adjusted_score:.1f}"
              f" repeat={seg.repeat_count} stutter={seg.stutter_count}{status}")

    # Step 8: Within-video duplicate detection
    print("\n🔄 Step 8: Duplicate content detection...")
    candidates = check_duplicate_content(candidates)
    print(f"   Marked {sum(1 for c in candidates if c.is_duplicate)} duplicate segments")

    # Step 9: Layered selection
    print("\n🏆 Step 9: Selecting best segment...")
    best = select_best_segment(candidates)
    if not best:
        print("   ❌ Cannot select best segment")
        return None

    print(f"   ✅ Best: segment_{best.index} | "
          f"{best.start_time:.1f}-{best.end_time:.1f}s ({best.duration:.1f}s) | "
          f"adjusted={best.adjusted_score:.1f} | natural_end={'yes' if best.is_natural_end else 'no'}")

    # Step 10: Clip output
    print(f"\n✂️  Step 10: Clipping output...")
    output_path = os.path.join(output_dir, f"{video_name}_clip.mp4")
    if not clip_segment(video_path, best, output_path):
        print(f"   ❌ Failed to clip")
        return None
    print(f"   ✅ Output: {output_path}")

    # Generate individual report only when not in batch mode
    if not batch_mode:
        report_path = os.path.join(output_dir, f"{video_name}_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {video_name} Clip Report\n\n")
            f.write(f"**Version**: v4.7\n")
            f.write(f"**Processed**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"## Video Info\n\n")
            f.write(f"- Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)\n")
            f.write(f"- Silence spans: {len(silences)}\n- Segments: {len(segments)}\n- Candidates: {len(candidates)}\n\n")
            f.write(f"## Candidate Comparison\n\n")
            f.write(f"| Segment | Time Range | Duration | Base | Adjusted | Natural End | Duplicate | Selected |\n")
            f.write(f"|---------|------------|----------|------|----------|-------------|-----------|----------|\n")
            for c in candidates:
                f.write(f"| seg_{c.index} | {c.start_time:.1f}-{c.end_time:.1f}s | "
                        f"{c.duration:.1f}s | {c.total_score} | {c.adjusted_score:.1f} | "
                        f"{'yes' if c.is_natural_end else 'no'} | "
                        f"{'yes' if c.is_duplicate else ''} | "
                        f"{'✅' if c.index == best.index else ''} |\n")
            f.write(f"\n## Final Selection\n\n")
            f.write(f"- **Segment**: segment_{best.index}\n")
            f.write(f"- **Time**: {best.start_time:.1f}s - {best.end_time:.1f}s\n")
            f.write(f"- **Duration**: {best.duration:.1f}s\n")
            f.write(f"- **Adjusted Score**: {best.adjusted_score:.1f}\n")
            if best.transcript:
                f.write(f"- **Transcript**: {best.transcript}\n")
        print(f"   📄 Report: {report_path}")

    return ClipInfo(
        video_name=video_name, clip_path=output_path,
        transcript=best.transcript, adjusted_score=best.adjusted_score,
        is_natural_end=best.is_natural_end, duration=best.duration,
    )

# --- Scenario B: Batch -> cross-video dedup -> concatenation ---
def process_batch(input_dir, output_dir, work_dir):
    """Scenario B: Output final concatenated video + single batch report"""
    video_files = sorted(
        glob.glob(os.path.join(input_dir, "*.MTS")) +
        glob.glob(os.path.join(input_dir, "*.mp4")) +
        glob.glob(os.path.join(input_dir, "*.mov"))
    )
    if not video_files:
        print("❌ No video files found"); return

    print(f"\n{'='*60}")
    print(f"  Video Auto Editor v4.7 - Scenario B (Batch)")
    print(f"  Input directory: {input_dir} ({len(video_files)} videos)")
    print(f"{'='*60}\n")

    # Phase 1: Process each video (batch_mode=True, no individual reports)
    clips = []
    for vf in video_files:
        clip = process_single_video(vf, output_dir, work_dir, batch_mode=True)
        if clip:
            clips.append(clip)

    if not clips:
        print("❌ No videos processed successfully"); return

    # Phase 2: Cross-video dedup
    print(f"\n{'='*60}")
    print(f"  🔄 Cross-video dedup check ({len(clips)} clips)")
    print(f"{'='*60}\n")

    clips = cross_video_dedup(clips)
    kept = [c for c in clips if not c.is_cross_duplicate]
    removed = [c for c in clips if c.is_cross_duplicate]

    for c in removed:
        print(f"   ❌ Remove {c.video_name} (duplicate of {c.duplicate_of}, adjusted {c.adjusted_score:.1f})")
    for c in kept:
        print(f"   ✅ Keep {c.video_name} (adjusted {c.adjusted_score:.1f})")
    print(f"\n   Dedup result: {len(clips)} -> {len(kept)} clips")

    # Phase 3: Concatenation
    print(f"\n{'='*60}")
    print(f"  🎬 Concatenating {len(kept)} clips")
    print(f"{'='*60}\n")

    final_path = os.path.join(output_dir, f"final_concat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mp4")
    if not concat_videos([c.clip_path for c in kept], final_path):
        print("   ❌ Concatenation failed"); return
    print(f"   ✅ Final video: {final_path}")

    # Phase 4: Clean intermediate files (individual clips + temp audio)
    for c in clips:
        if os.path.exists(c.clip_path):
            os.remove(c.clip_path)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)

    # Phase 5: Generate single batch report
    report_path = os.path.join(output_dir, "batch_report.md")
    total_dur = sum(c.duration for c in kept)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Batch Processing Report\n\n")
        f.write(f"**Version**: v4.7\n")
        f.write(f"**Processed**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        if removed:
            f.write(f"## Cross-Video Dedup\n\n")
            f.write(f"| Video | Adjusted | Natural End | Decision | Reason |\n")
            f.write(f"|-------|----------|--------------|----------|--------|\n")
            for c in clips:
                decision = "❌ Remove" if c.is_cross_duplicate else "✅ Keep"
                reason = f"duplicate of {c.duplicate_of}" if c.is_cross_duplicate else ""
                f.write(f"| {c.video_name} | {c.adjusted_score:.1f} | "
                        f"{'yes' if c.is_natural_end else 'no'} | {decision} | {reason} |\n")
            f.write(f"\n")

        f.write(f"## Final Concatenation ({len(kept)} clips)\n\n")
        f.write(f"| # | Video | Duration | Adjusted | Natural End | Transcript Summary |\n")
        f.write(f"|---|-------|----------|----------|-------------|--------------------|\n")
        for i, c in enumerate(kept, 1):
            summary = (c.transcript[:40] + "...") if c.transcript and len(c.transcript) > 40 else (c.transcript or "—")
            f.write(f"| {i} | {c.video_name} | {c.duration:.1f}s | "
                    f"{c.adjusted_score:.1f} | {'yes' if c.is_natural_end else 'no'} | {summary} |\n")
        f.write(f"\n**Total duration**: {total_dur:.1f}s ({total_dur/60:.1f}min)\n")
        f.write(f"\n**Output file**: `{final_path}`\n")

    print(f"   📄 Report: {report_path}")
    print(f"\n{'='*60}")
    print(f"  Batch processing complete! ({len(kept)}/{len(clips)} clips)")
    print(f"{'='*60}\n")

# --- Main entry ---
def main():
    if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
        work_dir = sys.argv[3] if len(sys.argv) > 3 else "./video_work"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
        process_batch(input_dir, output_dir, work_dir)
        return

    video_path = sys.argv[1] if len(sys.argv) > 1 else "02047.MTS"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    work_dir = "./video_work"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    clip = process_single_video(video_path, output_dir, work_dir)
    if clip:
        print(f"  Scenario A complete: {clip.clip_path}")
    else:
        print("  ❌ Processing failed")

if __name__ == "__main__":
    main()
