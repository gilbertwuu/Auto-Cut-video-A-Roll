#!/usr/bin/env python3
"""
视频自动剪辑系统 v4.7 - Video Auto Editor
Copyright (C) 2026

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY. See the LICENSE file for details.

场景A: 单视频粗剪（静音检测 → 段落识别 → 评分 → 转录 → 流畅度分析 → 去重 → 剪辑）
场景B: 批量粗剪 → 跨视频去重 → 拼接

依赖: FFmpeg, openai-whisper
"""
import subprocess, re, os, sys, json, glob, difflib, datetime, shutil
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- 配置 ---
CONFIG = {
    "silence_noise": -30,           # dB, 越小越严格
    "silence_duration": 0.8,        # 秒, 静音最小时长
    "min_score": 90,                # 基础分最低要求 (满分100)
    "min_duration": 15,             # 段落最低时长 (秒)
    "buffer_start": 1,              # 开始前缓冲 (秒)
    "buffer_end": 3,                # 结束后缓冲 (秒)
    "crf": 18,                      # 视频质量 (18=视觉无损)
    "preset": "fast",               # 编码速度
    "audio_bitrate": "192k",        # 音频码率
    "penalty_repeat": 5,            # 每次重复扣分
    "penalty_stutter": 3,           # 每次卡顿扣分
    "penalty_interrupt": 10,        # 突然中断扣分
    "bonus_natural_end": 5,         # 自然结束加分
    "bonus_completeness_max": 3,    # 完整度加分上限
    "duplicate_threshold": 0.7,     # 内容相似度阈值
}

# --- 数据结构 ---
@dataclass
class Segment:
    """视频段落"""
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
    """单个视频的粗剪结果，用于跨视频去重"""
    video_name: str
    clip_path: str
    transcript: str
    adjusted_score: float
    is_natural_end: bool
    duration: float
    is_cross_duplicate: bool = False
    duplicate_of: str = ""

# --- 模块1: 静音检测 ---
def detect_silence(video_path):
    """使用 FFmpeg silencedetect 滤镜检测静音段"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={CONFIG['silence_noise']}dB:d={CONFIG['silence_duration']}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    starts = re.findall(r'silence_start: ([\d.]+)', result.stderr)
    ends = re.findall(r'silence_end: ([\d.]+)', result.stderr)
    return [(float(starts[i]), float(ends[i])) for i in range(min(len(starts), len(ends)))]

# --- 模块2: 段落识别 ---
def identify_segments(silences, total_duration):
    """根据静音段将视频切分为段落（≥1秒的非静音区间）"""
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

# --- 模块3: 评分系统 (4维 × 25分 = 100分) ---
def _score_boundary(silences, time_point, total_duration, is_start):
    """评估段落边界的清晰度 (开始或结束), 返回 0-25 分"""
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
    """4维度评分: 清晰开始/结束 + 中间流畅 + 节奏自然"""
    seg.score_start = _score_boundary(silences, seg.start_time, total_duration, is_start=True)
    seg.score_end = _score_boundary(silences, seg.end_time, total_duration, is_start=False)

    # 中间流畅 (25分)
    internal = [s for s in silences
                if s[0] > seg.start_time + 0.1 and s[1] < seg.end_time - 0.1]
    seg.internal_silences = internal
    seg.interruption_count = len(internal)
    seg.interruption_duration = sum(e - s for s, e in internal)

    if seg.interruption_count == 0:      seg.score_fluency = 25
    elif seg.interruption_count <= 2:    seg.score_fluency = 20
    elif seg.interruption_count <= 4:    seg.score_fluency = 15
    else: seg.score_fluency = max(5, 25 - seg.interruption_count * 3)

    # 节奏自然 (25分): 停顿占比 (15分) + 最大单次停顿 (10分) + 极短片段惩罚
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

# --- 模块4: 转录 (Whisper CLI) ---
def transcribe_segment(video_path, seg, work_dir):
    """提取段落音频并调用 Whisper 转录为文本"""
    audio_path = os.path.join(work_dir, f"segment_{seg.index}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(seg.start_time), "-to", str(seg.end_time),
        "-vn", "-ar", "16000", "-ac", "1", audio_path
    ], capture_output=True, text=True)

    if not os.path.exists(audio_path):
        print(f"    ⚠️ 音频提取失败: segment_{seg.index}")
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
        print(f"    ⚠️ 转录文件未生成: segment_{seg.index}")
        return ""
    except Exception as e:
        print(f"    ⚠️ 转录失败: segment_{seg.index}: {e}")
        return ""

# --- 模块5: 流畅度分析 ---
def analyze_fluency(transcript):
    """分析转录文本，返回 (repeat_count, stutter_count, is_natural_end, is_interrupted)"""
    if not transcript:
        return 0, 0, False, False

    text = re.sub(r'(?i)\bwhisper\b', '', transcript.strip()).strip()

    # 滑动窗口短语重复检测 (2-4字短语在10字窗口内重复)
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

    # 卡顿检测
    stutter_count = sum(len(re.findall(p, text)) for p in [r'[嗯啊呃]', r'那个', r'就是说', r'\.{2,}', r'…'])

    # 突然中断检测 (20种连接词/未完成标记)
    interrupt_re = r'(的时候|然后|但是|如果|因为|而且|所以|就是|其实|那么|或者|并且|还是|不过|包括|比如说|另外|接下来|还有就是|就是说)$'
    is_interrupted = bool(re.search(interrupt_re, text))

    # 自然结束检测
    has_punctuation = bool(re.search(r'[。！？]$', text))
    is_connective_end = bool(re.search(interrupt_re, text))
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

# --- 模块6: 调整分计算 ---
def calculate_adjusted_score(seg):
    """在基础分上叠加流畅度惩罚/奖励，归一化到 0-100"""
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

# --- 模块7: 内容去重 (通用分组策略) ---
def _find_duplicate_groups(items, get_text):
    """通用去重分组: 两两比较文本相似度，相似的归为一组"""
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
    """视频内去重: 候选段落分组，组内保留最优，其余标记为重复"""
    groups = _find_duplicate_groups(candidates, lambda s: s.transcript)
    for group in groups:
        for i, j in [(i, j) for i in group for j in group if i < j]:
            print(f"    ⚠️  segment_{candidates[i].index} 和 segment_{candidates[j].index} 内容相似")
        best = max(group, key=lambda idx: (candidates[idx].is_natural_end, candidates[idx].adjusted_score, candidates[idx].index))
        for idx in group:
            if idx != best:
                candidates[idx].is_duplicate = True
                candidates[idx].duplicate_with.append(candidates[best].index)
    return candidates

def cross_video_dedup(clips):
    """跨视频去重: 对比各粗剪片段的转录内容，标记重复片段"""
    if len(clips) < 2:
        return clips
    groups = _find_duplicate_groups(clips, lambda c: c.transcript)
    for group in groups:
        for i, j in [(i, j) for i in group for j in group if i < j]:
            print(f"    ⚠️  {clips[i].video_name} 和 {clips[j].video_name} 内容相似")
        best = max(group, key=lambda idx: (clips[idx].is_natural_end, clips[idx].adjusted_score, clips[idx].video_name))
        for idx in group:
            if idx != best:
                clips[idx].is_cross_duplicate = True
                clips[idx].duplicate_of = clips[best].video_name
    return clips

# --- 模块8: 分层筛选 ---
def select_best_segment(candidates):
    """分层筛选: 自然结束 → 流畅度 → 调整分 → 时长/索引"""
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

# --- 模块9: FFmpeg 操作 ---
def clip_segment(video_path, seg, output_path):
    """使用 FFmpeg 剪辑目标段落 (带前后缓冲)"""
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
    """使用 FFmpeg concat 协议拼接视频列表"""
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

# --- 场景A: 单视频处理 ---
def process_single_video(video_path, output_dir, work_dir, batch_mode=False):
    """
    处理单个视频，返回 ClipInfo；若失败返回 None。
    batch_mode=True 时不生成单独报告（由场景B统一报告）。
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_work = os.path.join(work_dir, video_name)
    os.makedirs(video_work, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  处理: {video_name}" if batch_mode else f"  视频自动剪辑系统 v4.7 - 场景A\n  输入: {video_path}")
    print(f"{'='*60}\n")

    # 步骤1: 获取视频信息
    print("📋 步骤1: 获取视频信息...")
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        total_duration = float(json.loads(result.stdout)['format']['duration'])
    except Exception as e:
        print(f"   ❌ 无法获取视频信息: {e}")
        return None
    print(f"   时长: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # 步骤2: 静音检测
    print("\n🔇 步骤2: 静音检测...")
    silences = detect_silence(video_path)
    print(f"   检测到 {len(silences)} 个静音段")

    # 步骤3: 段落识别
    print("\n📝 步骤3: 段落识别...")
    segments = identify_segments(silences, total_duration)
    print(f"   识别到 {len(segments)} 个段落")

    # 步骤4: 评分
    print("\n⭐ 步骤4: 评分...")
    for seg in segments:
        score_segment(seg, silences, total_duration)
        print(f"   segment_{seg.index}: {seg.start_time:.1f}s-{seg.end_time:.1f}s "
              f"({seg.duration:.1f}s) 分数={seg.total_score}")

    # 步骤5: 筛选候选
    print(f"\n🔍 步骤5: 筛选候选 (min_score={CONFIG['min_score']}, min_duration={CONFIG['min_duration']}s)...")
    candidates = [s for s in segments if s.total_score >= CONFIG['min_score'] and s.duration >= CONFIG['min_duration']]
    print(f"   {len(candidates)} 个候选段落")

    if not candidates:
        print("\n⚠️  没有满足条件的候选段落，降低标准...")
        candidates = sorted(segments, key=lambda s: s.total_score, reverse=True)[:5]
        print(f"   选择了评分最高的 {len(candidates)} 个段落")

    # 步骤6: 转录
    print("\n🎤 步骤6: 转录候选段落...")
    whisper_available = True
    try:
        if subprocess.run(["python3", "-m", "whisper", "--help"], capture_output=True, text=True, timeout=10).returncode != 0:
            whisper_available = False
    except Exception:
        whisper_available = False

    if whisper_available:
        for seg in candidates:
            print(f"   转录 segment_{seg.index}...")
            seg.transcript = transcribe_segment(video_path, seg, video_work)
            if seg.transcript:
                preview = seg.transcript[:50] + "..." if len(seg.transcript) > 50 else seg.transcript
                print(f"   ✅ [{preview}]")
    else:
        print("   ⚠️  Whisper未安装，跳过转录，使用纯音频评分")

    # 步骤7: 流畅度分析
    print("\n📊 步骤7: 流畅度分析...")
    for seg in candidates:
        if seg.transcript:
            seg.repeat_count, seg.stutter_count, seg.is_natural_end, seg.is_interrupted = analyze_fluency(seg.transcript)
        seg.adjusted_score = calculate_adjusted_score(seg)
        status = (" ✅自然结束" if seg.is_natural_end else "") + (" ❌中断" if seg.is_interrupted else "")
        print(f"   segment_{seg.index}: 基础={seg.total_score} 调整={seg.adjusted_score:.1f}"
              f" 重复={seg.repeat_count} 卡顿={seg.stutter_count}{status}")

    # 步骤8: 视频内重复检测
    print("\n🔄 步骤8: 重复内容检测...")
    candidates = check_duplicate_content(candidates)
    print(f"   标记了 {sum(1 for c in candidates if c.is_duplicate)} 个重复段落")

    # 步骤9: 分层筛选
    print("\n🏆 步骤9: 分层筛选最佳段落...")
    best = select_best_segment(candidates)
    if not best:
        print("   ❌ 无法选择最佳段落")
        return None

    print(f"   ✅ 最佳: segment_{best.index} | "
          f"{best.start_time:.1f}-{best.end_time:.1f}s ({best.duration:.1f}s) | "
          f"调整分={best.adjusted_score:.1f} | 自然结束={'是' if best.is_natural_end else '否'}")

    # 步骤10: 剪辑输出
    print(f"\n✂️  步骤10: 剪辑输出...")
    output_path = os.path.join(output_dir, f"{video_name}_粗剪.mp4")
    if not clip_segment(video_path, best, output_path):
        print(f"   ❌ 剪辑失败")
        return None
    print(f"   ✅ 输出: {output_path}")

    # 场景A独立运行时才生成单独报告
    if not batch_mode:
        report_path = os.path.join(output_dir, f"{video_name}_报告.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {video_name} 粗剪报告\n\n")
            f.write(f"**系统版本**: v4.7\n")
            f.write(f"**处理时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"## 视频信息\n\n")
            f.write(f"- 时长: {total_duration:.1f}s ({total_duration/60:.1f}min)\n")
            f.write(f"- 静音段: {len(silences)}个\n- 识别段落: {len(segments)}个\n- 候选段落: {len(candidates)}个\n\n")
            f.write(f"## 候选段落对比\n\n")
            f.write(f"| 段落 | 时间范围 | 时长 | 基础分 | 调整分 | 自然结束 | 重复 | 选中 |\n")
            f.write(f"|------|---------|------|--------|--------|---------|------|------|\n")
            for c in candidates:
                f.write(f"| seg_{c.index} | {c.start_time:.1f}-{c.end_time:.1f}s | "
                        f"{c.duration:.1f}s | {c.total_score} | {c.adjusted_score:.1f} | "
                        f"{'是' if c.is_natural_end else '否'} | "
                        f"{'重复' if c.is_duplicate else ''} | "
                        f"{'✅' if c.index == best.index else ''} |\n")
            f.write(f"\n## 最终选择\n\n")
            f.write(f"- **段落**: segment_{best.index}\n")
            f.write(f"- **时间**: {best.start_time:.1f}s - {best.end_time:.1f}s\n")
            f.write(f"- **时长**: {best.duration:.1f}s\n")
            f.write(f"- **调整分**: {best.adjusted_score:.1f}\n")
            if best.transcript:
                f.write(f"- **转录内容**: {best.transcript}\n")
        print(f"   📄 报告: {report_path}")

    return ClipInfo(
        video_name=video_name, clip_path=output_path,
        transcript=best.transcript, adjusted_score=best.adjusted_score,
        is_natural_end=best.is_natural_end, duration=best.duration,
    )

# --- 场景B: 批量处理 → 跨视频去重 → 拼接 ---
def process_batch(input_dir, output_dir, work_dir):
    """场景B: 只输出最终拼接视频 + 一份批量报告"""
    video_files = sorted(
        glob.glob(os.path.join(input_dir, "*.MTS")) +
        glob.glob(os.path.join(input_dir, "*.mp4")) +
        glob.glob(os.path.join(input_dir, "*.mov"))
    )
    if not video_files:
        print("❌ 未找到视频文件"); return

    print(f"\n{'='*60}")
    print(f"  视频自动剪辑系统 v4.7 - 场景B (批量)")
    print(f"  输入目录: {input_dir} ({len(video_files)} 个视频)")
    print(f"{'='*60}\n")

    # 阶段1: 逐个粗剪 (batch_mode=True, 不生成单独报告)
    clips = []
    for vf in video_files:
        clip = process_single_video(vf, output_dir, work_dir, batch_mode=True)
        if clip:
            clips.append(clip)

    if not clips:
        print("❌ 没有成功处理的视频"); return

    # 阶段2: 跨视频去重
    print(f"\n{'='*60}")
    print(f"  🔄 跨视频去重检查 ({len(clips)} 个粗剪片段)")
    print(f"{'='*60}\n")

    clips = cross_video_dedup(clips)
    kept = [c for c in clips if not c.is_cross_duplicate]
    removed = [c for c in clips if c.is_cross_duplicate]

    for c in removed:
        print(f"   ❌ 移除 {c.video_name} (与 {c.duplicate_of} 重复, 调整分 {c.adjusted_score:.1f})")
    for c in kept:
        print(f"   ✅ 保留 {c.video_name} (调整分 {c.adjusted_score:.1f})")
    print(f"\n   去重结果: {len(clips)} → {len(kept)} 个片段")

    # 阶段3: 拼接
    print(f"\n{'='*60}")
    print(f"  🎬 拼接 {len(kept)} 个粗剪片段")
    print(f"{'='*60}\n")

    final_path = os.path.join(output_dir, f"最终拼接_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mp4")
    if not concat_videos([c.clip_path for c in kept], final_path):
        print("   ❌ 拼接失败"); return
    print(f"   ✅ 最终视频: {final_path}")

    # 阶段4: 清理中间文件 (单个粗剪视频 + 临时音频)
    for c in clips:
        if os.path.exists(c.clip_path):
            os.remove(c.clip_path)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)

    # 阶段5: 生成唯一一份报告
    report_path = os.path.join(output_dir, "批量处理报告.md")
    total_dur = sum(c.duration for c in kept)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 视频批量剪辑报告\n\n")
        f.write(f"**系统版本**: v4.7\n")
        f.write(f"**处理时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        if removed:
            f.write(f"## 跨视频去重\n\n")
            f.write(f"| 视频 | 调整分 | 自然结束 | 决策 | 原因 |\n")
            f.write(f"|------|-------|---------|------|------|\n")
            for c in clips:
                decision = "❌ 移除" if c.is_cross_duplicate else "✅ 保留"
                reason = f"与 {c.duplicate_of} 重复" if c.is_cross_duplicate else ""
                f.write(f"| {c.video_name} | {c.adjusted_score:.1f} | "
                        f"{'是' if c.is_natural_end else '否'} | {decision} | {reason} |\n")
            f.write(f"\n")

        f.write(f"## 最终拼接 ({len(kept)} 个片段)\n\n")
        f.write(f"| 序号 | 视频 | 时长 | 调整分 | 自然结束 | 转录摘要 |\n")
        f.write(f"|------|------|------|-------|----------|----------|\n")
        for i, c in enumerate(kept, 1):
            summary = (c.transcript[:40] + "...") if c.transcript and len(c.transcript) > 40 else (c.transcript or "—")
            f.write(f"| {i} | {c.video_name} | {c.duration:.1f}s | "
                    f"{c.adjusted_score:.1f} | {'是' if c.is_natural_end else '否'} | {summary} |\n")
        f.write(f"\n**总时长**: {total_dur:.1f}s ({total_dur/60:.1f}min)\n")
        f.write(f"\n**输出文件**: `{final_path}`\n")

    print(f"   📄 报告: {report_path}")
    print(f"\n{'='*60}")
    print(f"  批量处理完成! ({len(kept)}/{len(clips)} 个片段)")
    print(f"{'='*60}\n")

# --- 主入口 ---
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
        print(f"  场景A完成: {clip.clip_path}")
    else:
        print("  ❌ 处理失败")

if __name__ == "__main__":
    main()
