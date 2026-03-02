#!/usr/bin/env python3
"""
凡人修仙傳 — 坠魔谷篇（宋玉被擒）AI 影片生成腳本
使用 Google Gemini API + Veo 3.1 生成場景影片
使用 edge-tts 生成中文配音
使用 ffmpeg 混音 + 串接

Usage:
    python fanren_zhuimo_video.py              # 生成全部場景
    python fanren_zhuimo_video.py --scene 3    # 只生成第 3 個場景
    python fanren_zhuimo_video.py --concat     # 只做串接
    python fanren_zhuimo_video.py --tts-only   # 只生成配音
    python fanren_zhuimo_video.py --list       # 列出場景
"""

import os
import sys
import time
import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from google import genai
from google.genai import types

# ── Config ──
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "veo-3.1-generate-preview"
OUTPUT_DIR = Path(__file__).parent / "fanren_zhuimo_clips"
FINAL_OUTPUT = Path(__file__).parent / "fanren_zhuimo_songyu.mp4"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Veo 3.1 settings
VIDEO_DURATION = 8        # seconds per clip
ASPECT_RATIO = "16:9"
RESOLUTION = "720p"

# ── 3D Donghua Style (matching 原力動畫 style) ──
STYLE = (
    "High-quality 3D CG Chinese donghua style, photorealistic semi-stylized rendering "
    "by Original Force animation studio. "
    "Chinese xianxia fantasy aesthetic, cinematic volumetric lighting, "
    "cool teal-blue shadows with warm amber highlights. "
    "Dramatic color grading, atmospheric fog and mist, "
    "glowing magical qi particles and energy effects. "
)

# ── TTS Voice Config ──
# edge-tts Chinese voices
VOICE_NARRATOR = "zh-CN-YunxiNeural"       # 男聲旁白
VOICE_HANLI = "zh-CN-YunjianNeural"         # 韓立 — 沉穩男聲
VOICE_SONGYU = "zh-CN-XiaoxiaoNeural"       # 宋玉 — 溫婉女聲
VOICE_MUPEILING = "zh-CN-XiaoyiNeural"      # 慕沛靈 — 柔和女聲
VOICE_LIUYU = "zh-CN-XiaoyiNeural"          # 柳玉 — 用不同語速區分
VOICE_SECOND_SOUL = "zh-CN-YunyangNeural"   # 第二元嬰 — 陰沉男聲

# ══════════════════════════════════════════════════════════════
# 坠魔谷篇 — 宋玉被擒 場景分鏡（約10分鐘 = 75場景 × 8秒）
# 精選核心劇情：三女入谷 → 被擒 → 韓立救援
# ══════════════════════════════════════════════════════════════

SCENES = [
    # ═══ ACT 1: 落雲宗 — 出發前 ═══
    {
        "id": 1,
        "title": "落雲宗全景 — 清晨",
        "prompt": (
            "Sweeping aerial shot of a magnificent Chinese xianxia sect built on floating mountain peaks "
            "above a sea of clouds at dawn. Multiple pagodas and cultivation halls connected by stone bridges. "
            "Golden sunrise light illuminates the misty peaks. Cranes fly between the mountains. "
            "Majestic and serene cultivation sect headquarters."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "落雲宗，天南修仙界第一大派。自從韓立晉入元嬰後期，已無人敢輕視此宗。",
    },
    {
        "id": 2,
        "title": "白鳳峰 — 宋玉的住所",
        "prompt": (
            "A beautiful peak covered in white phoenix flowers and bamboo groves. An elegant pavilion "
            "sits at the summit with flowing water features. A stunningly beautiful woman in blue robes "
            "with clear bright eyes and ornamental hairpins stands at the pavilion railing, looking at "
            "a jade slip in her hand with a thoughtful expression. Morning light creates a halo effect. "
            "Serene cultivation aesthetic."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "幻靈草⋯⋯此草生於坠魔谷內谷深處，若能得到，對師姐的築基丹大有裨益。",
    },
    {
        "id": 3,
        "title": "宋玉召集 — 議事廳",
        "prompt": (
            "Interior of an elegant Chinese cultivation hall with wooden pillars and silk curtains. "
            "Three women sit around a circular jade table. A beautiful woman in blue robes (Song Yu) "
            "sits at the head, gesturing as she speaks. A woman in purple robes (Mu Peiling) listens "
            "attentively. A tall woman in dark green robes (Liu Yu) leans forward with interest. "
            "Warm interior candlelight, silk scrolls on walls."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "紫靈師姐給了我一份內谷的安全路線圖。我們三人結伴而行，應可一試。",
    },
    {
        "id": 4,
        "title": "慕沛靈的擔憂",
        "prompt": (
            "Close-up of a pretty woman in light purple robes with worried eyes, clasping her hands. "
            "Warm candlelight from the side. She bites her lip slightly, looking between the other "
            "two women. Soft focus background of the cultivation hall interior."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "內谷太過危險了⋯⋯韓前輩曾說過，沒有元嬰期修為，不宜深入。",
    },
    {
        "id": 5,
        "title": "柳玉的自信",
        "prompt": (
            "A tall attractive woman in dark green robes with sharp features smiles confidently, "
            "raising one hand where a faint frost-blue glow emanates — hint of her spirit insects. "
            "Her expression shows calculated confidence. She stands near a window with mountain view."
        ),
        "voice": VOICE_LIUYU,
        "dialogue": "我的六翼霜蚣已達四級巔峰，足以應對谷中大部分妖獸。宋師姐放心。",
    },
    {
        "id": 6,
        "title": "宋玉做決定 — 出發",
        "prompt": (
            "The beautiful woman in blue robes stands decisively, her clear bright eyes showing "
            "determination. She rolls up a jade map and tucks it into her sleeve. Behind her through "
            "the window, distant mountains shrouded in colorful miasma are visible. "
            "Golden hour lighting, dramatic composition."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "就這麼定了。明日一早，我們出發前往坠魔谷。只取幻靈草，不做停留。",
    },

    # ═══ ACT 2: 坠魔谷外圍 ═══
    {
        "id": 7,
        "title": "萬嶺山脈 — 遠景",
        "prompt": (
            "Dramatic wide shot of a vast mountain range shrouded in multicolored toxic miasma — "
            "purple, green, and yellow poisonous fog swirling between dark jagged peaks. The atmosphere "
            "is ominous and foreboding. Three small flying light streaks (sword riders) approach the "
            "mountain range from the distance. Dramatic clouds, dark color palette."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "萬嶺山脈，坠魔谷所在之地。此處瘴氣常年不散，被稱為天南第一凶地。",
    },
    {
        "id": 8,
        "title": "三女飛行 — 穿越瘴氣",
        "prompt": (
            "Three women flying on glowing swords through swirling purple-green miasma between dark "
            "mountain peaks. The woman in blue (lead) projects a translucent white barrier shield "
            "around all three. Toxic fog parts around the barrier. Green energy particles scatter. "
            "Dynamic flight scene, speed lines, wind effects."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "屏住呼吸，跟緊我的護體靈光。這層瘴氣毒性不弱。",
    },
    {
        "id": 9,
        "title": "坠魔谷入口 — 外谷",
        "prompt": (
            "A massive dark valley entrance between two towering cliff faces, carved with ancient "
            "runes that faintly glow. Wisps of dark energy seep from the entrance. The three women "
            "land at the entrance, looking up at the imposing gateway. Scale shows how small they "
            "are compared to the enormous valley entrance. Ominous atmosphere."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "坠魔谷，上古修士大戰的遺跡。空間裂縫密佈，凶險萬分。",
    },
    {
        "id": 10,
        "title": "外谷搜尋 — 荒涼景色",
        "prompt": (
            "Wide shot of a desolate ancient valley with twisted dead trees, scattered bones of "
            "long-dead cultivators, and faint spatial distortions shimmering in the air like heat haze. "
            "Three women carefully walk through, checking beneath rocks and in crevices. "
            "Eerie green-blue ambient light, mist along the ground."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "這裡什麼都沒有⋯⋯只有死氣和空間裂縫的殘留。",
    },
    {
        "id": 11,
        "title": "外谷探索 — 一天一夜",
        "prompt": (
            "Time-lapse montage style: three women searching the outer valley. Day turns to night "
            "turns to day. They check cave entrances, climb ridges, examine glowing plants. "
            "Their expressions gradually shift from hopeful to disappointed. "
            "The woman in blue (Song Yu) consults her jade map repeatedly."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "三人在外谷搜尋了一天一夜，卻始終未能找到幻靈草的蹤跡。",
    },
    {
        "id": 12,
        "title": "宋玉查看路線圖",
        "prompt": (
            "Close-up of the beautiful woman in blue robes holding a glowing jade slip, her face "
            "illuminated by its ethereal light. A translucent 3D map hovers above the slip showing "
            "valley terrain with a glowing path marked into the inner valley. She traces the path "
            "with her finger, expression conflicted. Night scene, firelight."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "紫靈師姐標記的位置⋯⋯都在內谷。看來外谷確實沒有幻靈草。",
    },

    # ═══ ACT 3: 進入內谷 ═══
    {
        "id": 13,
        "title": "內谷入口 — 抉擇",
        "prompt": (
            "Three women stand before a narrow passage between black obsidian cliffs. Beyond the "
            "passage, the air shimmers with violent spatial distortions and flashes of lightning-like "
            "energy. A sense of overwhelming danger emanates from the inner valley. "
            "The woman in blue looks at the passage with conflicted determination."
        ),
        "voice": VOICE_LIUYU,
        "dialogue": "宋師姐，真的要進去嗎？這裡的靈氣波動⋯⋯比外谷強了十倍不止。",
    },
    {
        "id": 14,
        "title": "宋玉的決心",
        "prompt": (
            "Close-up of the beautiful woman in blue, her clear bright eyes hardening with resolve. "
            "Wind whips her hair and robes. She clutches the jade map tightly. "
            "Behind her, the other two women wait for her decision. "
            "Dramatic backlighting from the spatial distortions ahead."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "進。紫靈師姐的路線圖既然標明了安全通道，我們按圖索驥，速去速回。",
    },
    {
        "id": 15,
        "title": "穿越空間裂縫帶",
        "prompt": (
            "The three women carefully navigate through a corridor of violent spatial rifts — "
            "glowing tears in reality that reveal glimpses of other dimensions. They follow a narrow "
            "safe path marked by the map, ducking under warping space and stepping over unstable ground. "
            "Rainbow prismatic light from the rifts, intense VFX."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "三人小心翼翼地穿過空間裂縫帶，每一步都可能萬劫不復。",
    },
    {
        "id": 16,
        "title": "內谷全景 — 異世界般的景觀",
        "prompt": (
            "Breathtaking wide shot of the inner valley — a surreal landscape with floating rock "
            "formations, bioluminescent plants glowing in alien colors, ancient broken pillars and "
            "ruins from the primordial battle. Crystals jut from the ground emitting soft light. "
            "The sky above is distorted, showing glimpses of multiple overlapping dimensions. "
            "Beautiful but deeply unsettling."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "這裡⋯⋯就像是另一個世界。",
    },
    {
        "id": 17,
        "title": "內谷搜尋 — 發光草藥",
        "prompt": (
            "The three women walk through a grove of bioluminescent plants in the inner valley. "
            "Strange glowing mushrooms and crystalline flowers surround them. The woman in dark green "
            "(Liu Yu) kneels to examine a plant, shaking her head — not the target herb. "
            "Ethereal blue-green lighting, alien beauty."
        ),
        "voice": VOICE_LIUYU,
        "dialogue": "又不是⋯⋯這裡的靈草雖多，但都不是幻靈草。",
    },
    {
        "id": 18,
        "title": "深入搜索 — 時間流逝",
        "prompt": (
            "Montage of the three women searching deeper in the inner valley. Checking beneath "
            "ancient ruins, near underground streams with glowing water, around crystal formations. "
            "Their movements become more urgent and fatigued. Tension builds in the atmosphere."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "三人又搜尋了整整一天一夜，將內谷翻了個遍，卻仍然一無所獲。",
    },
    {
        "id": 19,
        "title": "宋玉下令撤退",
        "prompt": (
            "The beautiful woman in blue robes raises her hand in a stop gesture, her expression "
            "pragmatic and calm despite the disappointment. She addresses the other two women firmly. "
            "Behind them, a faint dark shadow moves in the distant fog, unnoticed. "
            "Ominous low-angle shot."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "算了，沒機緣強求不來的。而且內谷過於危險，我們立刻撤退。",
    },
    {
        "id": 20,
        "title": "慕沛靈鬆一口氣",
        "prompt": (
            "Close-up of the woman in purple robes sighing with relief, her shoulders dropping. "
            "She nods quickly in agreement. A small grateful smile appears on her worried face. "
            "Warm lighting on her face contrasting with the eerie valley background."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "宋師姐說得對。平安回去才是最重要的。",
    },

    # ═══ ACT 4: 追殺 — 黑雲降臨 ═══
    {
        "id": 21,
        "title": "黑雲出現 — 天色驟變",
        "prompt": (
            "Dramatic sky change above the inner valley. A massive swirling black cloud, alive with "
            "dark lightning and demonic energy, suddenly materializes overhead. The three women look up "
            "in horror. The temperature drops, frost forming on nearby crystals. "
            "Extreme dramatic lighting shift from eerie calm to terrifying darkness."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "就在三人準備離開之際，一股恐怖的妖氣從谷底深處暴涌而出。",
    },
    {
        "id": 22,
        "title": "宋玉感知危險",
        "prompt": (
            "Close-up of Song Yu's face, her clear eyes widening in absolute terror. Her spiritual "
            "sense (shown as translucent waves emanating from her forehead) detects something "
            "overwhelmingly powerful. Her pupils contract. She grabs the arms of both other women. "
            "Extreme emotional close-up, fear and urgency."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "不好！元嬰期的氣息！快跑！！",
    },
    {
        "id": 23,
        "title": "宋玉釋放白色圓珠",
        "prompt": (
            "The woman in blue robes throws a white glowing pearl into the air. The pearl expands "
            "rapidly, transforming into a massive white cloud platform large enough for three people. "
            "Brilliant white light erupts as the artifact activates. She pushes the other two women "
            "onto the cloud. Intense action scene, bright VFX against dark sky."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "快上來！這是我宋家的逃命法寶！",
    },
    {
        "id": 24,
        "title": "白雲急速飛行",
        "prompt": (
            "The three women ride the white cloud at incredible speed through the inner valley, "
            "weaving between floating rocks and crystal formations. Wind tears at their robes and hair. "
            "Behind them, the massive black cloud pursues relentlessly, gaining ground. "
            "High-speed chase scene, motion blur, wind effects."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "宋玉祭出家傳法寶，化為白雲急速逃離。然而身後的黑影，速度更在其上。",
    },
    {
        "id": 25,
        "title": "黑雲逼近 — 壓迫感",
        "prompt": (
            "Rear view of the white cloud fleeing, with the enormous black cloud looming ever closer "
            "behind. Dark tendrils of demonic energy reach forward like grasping fingers. "
            "The black cloud dwarfs the white one, filling the entire background. "
            "Terrifying scale difference, overwhelming darkness."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "追上來了！！越來越近了！！",
    },
    {
        "id": 26,
        "title": "柳玉放出霜蚣",
        "prompt": (
            "The tall woman in dark green robes turns to face the pursuing darkness, her eyes fierce "
            "and determined. She waves her hand and dozens of large frost-blue centipede-like spirit "
            "insects burst from her sleeve, forming a defensive wall of frost and ice energy "
            "between the white cloud and the pursuer. Blue ice particles scatter."
        ),
        "voice": VOICE_LIUYU,
        "dialogue": "我來斷後！六翼霜蚣，結陣！",
    },
    {
        "id": 27,
        "title": "霜蚣被瞬殺 — 黑色火焰",
        "prompt": (
            "A wave of terrifying black flames erupts from the pursuing dark cloud, engulfing the "
            "frost centipede formation entirely. The ice-blue glow is swallowed by black fire in "
            "an instant. The spirit insects disintegrate, their frozen energy evaporating. "
            "Shocking power disparity. Black flames vs blue frost, dramatic VFX."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "然而那黑色火焰瞬間便將數十頭四級巔峰的靈蟲化為灰燼。",
    },
    {
        "id": 28,
        "title": "柳玉的震驚",
        "prompt": (
            "Close-up of Liu Yu's face transitioning from fierce determination to utter shock. "
            "Her mouth falls open, color draining from her face. The black flames reflect in her "
            "wide horrified eyes. Behind her, the other two women share the same horror. "
            "Emotional devastation, slow-motion feel."
        ),
        "voice": VOICE_LIUYU,
        "dialogue": "怎麼可能⋯⋯四級巔峰的靈蟲，竟然一招就⋯⋯",
    },
    {
        "id": 29,
        "title": "巨大黑手 — 抓捕",
        "prompt": (
            "A massive shadowy black hand, formed of concentrated demonic qi, reaches out from the "
            "dark cloud and grabs the white cloud platform. The three women scream as the white cloud "
            "cracks and shatters under the grip. Dark energy wraps around them like chains. "
            "Terrifying capture scene, overwhelming dark power vs helpless victims."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "一只由魔氣凝聚的巨手，輕易地將白雲法寶捏碎，將三人盡數擒獲。",
    },
    {
        "id": 30,
        "title": "三女墜落 — 失去意識",
        "prompt": (
            "The three women fall through the air, bound by tendrils of black demonic energy, "
            "their eyes closing as consciousness fades. Their robes flutter in the fall. "
            "The camera follows them downward into darkness. "
            "Slow-motion fall, fading to black, somber and tragic."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "韓⋯⋯前輩⋯⋯",
    },

    # ═══ ACT 5: 洞穴囚禁 — 真相揭曉 ═══
    {
        "id": 31,
        "title": "洞穴 — 宋玉醒來",
        "prompt": (
            "First-person waking shot inside a dark cave lit by glowing stalactites. Perspective "
            "gradually focuses. Stone walls covered in glowing restriction runes. "
            "A beautiful woman in torn blue robes (Song Yu) struggles to move but finds herself "
            "paralyzed by magical restrictions — translucent glowing chains bind her limbs."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "不知過了多久，宋玉緩緩睜開雙眼，發現自己被禁制困在一處幽暗的洞穴之中。",
    },
    {
        "id": 32,
        "title": "三人被囚 — 禁制陣法",
        "prompt": (
            "Wide shot of the cave interior. Three women are bound to separate crystal pillars "
            "by glowing restriction arrays — complex circular magic formations on the ground. "
            "Song Yu (blue robes, center), Mu Peiling (purple, left), Liu Yu (green, right). "
            "All conscious but unable to move. Eerie green-purple cave lighting."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "宋師姐！你醒了！我們⋯⋯我們被人抓了。",
    },
    {
        "id": 33,
        "title": "黑影出現 — 腳步聲",
        "prompt": (
            "Dark corridor of the cave. Heavy footsteps echo. A shadowy figure approaches from "
            "the darkness, wreathed in swirling black demonic qi. Only the silhouette is visible — "
            "a tall imposing figure with an unnaturally shaped head. "
            "The three women's faces show pure terror as the shadow grows larger."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "沉重的腳步聲在洞穴中迴盪。一個令人膽寒的身影，從黑暗深處走來。",
    },
    {
        "id": 34,
        "title": "第二元嬰現身 — 骷髏面容",
        "prompt": (
            "Dramatic reveal shot. The figure steps into the cave light — a terrifying humanoid "
            "with a skull-like head, protruding fangs, grey-black corpse skin. One arm gleams "
            "gold-and-silver metallic. Black demonic qi swirls around him like living smoke. "
            "His eyes — eerily human, eerily familiar — contrast with the monstrous face. "
            "Horror reveal, dramatic lighting from below."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "那是一個骷髏般的面容，獠牙外露，一條手臂閃爍著金銀之色。渾身散發著令人窒息的玄陰魔氣。",
    },
    {
        "id": 35,
        "title": "柳玉驚恐尖叫",
        "prompt": (
            "Close-up of Liu Yu screaming in terror at the sight of the demonic figure. "
            "Her composure completely shattered. She struggles against her bindings uselessly. "
            "The demonic figure's shadow falls across her. Extreme horror reaction."
        ),
        "voice": VOICE_LIUYU,
        "dialogue": "魔⋯⋯魔修！！是元嬰期的魔修！！",
    },
    {
        "id": 36,
        "title": "宋玉冷靜觀察 — 認出眼神",
        "prompt": (
            "Split screen / close-up comparison. Song Yu's clear eyes studying the monster carefully "
            "despite her fear. She focuses on the creature's eyes — a flash of recognition crosses "
            "her face. Cut to the creature's eyes: despite the skull face, they hold intelligence "
            "and a familiar cold calm. A memory flash of Han Li's eyes overlays."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "等等⋯⋯這雙眼睛。我見過這雙眼睛。這是⋯⋯韓前輩的眼神？",
    },
    {
        "id": 37,
        "title": "第二元嬰說話",
        "prompt": (
            "The demonic skull-faced figure speaks, his voice distorted but carrying Han Li's "
            "speech patterns. He gestures with his metallic arm. Behind him, more black qi swirls "
            "forming vague shapes of power. The three women listen in mixed terror and confusion. "
            "Dark cave interior, dramatic close-up of the fanged mouth moving."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "韓立？哼⋯⋯我確實是從他身上分離出來的。但我，才是真正的韓立。",
    },
    {
        "id": 38,
        "title": "慕沛靈認出 — 玉佩感應",
        "prompt": (
            "Mu Peiling's eyes widen as a jade pendant around her neck suddenly glows and vibrates "
            "intensely. She looks down at it, then back at the demonic figure with dawning "
            "comprehension. The pendant emits a spiritual light that resonates with the creature's aura. "
            "Revelation moment, glowing artifact, emotional shock."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "這塊玉佩⋯⋯銀月姐姐給我的玉佩在感應他的氣息！他真的是⋯⋯第二元嬰！",
    },
    {
        "id": 39,
        "title": "第二元嬰的野心",
        "prompt": (
            "The demonic figure paces before the three imprisoned women, monologuing. "
            "His metallic arm catches the cave light as he gestures grandly. "
            "Behind him, shadows on the wall form an eerie mirror of Han Li's silhouette — "
            "the duality of his nature. Dark dramatic lighting, villain reveal."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "我在這坠魔谷中修煉多年，已佔據天煞魔尸之軀。等我吞噬了韓立的本體元嬰⋯⋯我，才是唯一的韓立。",
    },
    {
        "id": 40,
        "title": "宋玉的怒意",
        "prompt": (
            "Close-up of Song Yu's face transitioning from shock to cold anger. Despite being bound "
            "and helpless, her eyes burn with defiance. She stares directly at the Second Nascent Soul "
            "without flinching. Her jaw clenches. A subtle glow of spiritual energy flickers around her "
            "even through the restrictions."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "你不是韓前輩。韓前輩絕不會傷害自己的同門。你只是一個⋯⋯失控的分身罷了。",
    },
    {
        "id": 41,
        "title": "第二元嬰怒視",
        "prompt": (
            "The skull-faced figure's expression twists with rage at Song Yu's defiance. "
            "His metallic arm clenches into a fist, cracking the stone floor. Black fire erupts "
            "around him briefly before he regains composure. He smirks coldly with his fangs showing. "
            "Intimidating villain moment."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "嘿⋯⋯大膽。等韓立來救你們的時候，就讓他看看，他的人，是怎麼落在我手裡的。",
    },
    {
        "id": 42,
        "title": "第二元嬰離去 — 洞穴封印",
        "prompt": (
            "The demonic figure turns and walks back into the dark corridor. As he passes the cave "
            "entrance, he waves his metallic arm and a massive dark barrier seals the cave with "
            "a complex demonic restriction array — glowing red and black runes. "
            "The three women are left in the dim cave, alone. Door sealing with dramatic VFX."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "第二元嬰設下重重禁制後離去，將三人困於洞穴之中，作為引誘韓立的誘餌。",
    },
    {
        "id": 43,
        "title": "三人被困 — 沉默",
        "prompt": (
            "Wide shot of the sealed cave. Three women bound to crystal pillars in the eerie glow "
            "of restriction arrays. Long silence. Song Yu closes her eyes, seemingly meditating. "
            "Mu Peiling silently weeps. Liu Yu stares at the sealed entrance with impotent rage. "
            "Somber atmosphere, despair and determination mixed."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "⋯⋯不要怕。韓前輩一定會來的。他一定會來。",
    },

    # ═══ ACT 6: 韓立震天南 ═══
    {
        "id": 44,
        "title": "落雲宗大典 — 萬修來朝",
        "prompt": (
            "Massive aerial shot of the Luoyun Sect with thousands of cultivators gathered on the "
            "main plaza. Flying swords and spiritual lights fill the sky. A grand ceremony platform "
            "at the center. Flags and banners flutter. An atmosphere of awe and reverence. "
            "Epic scale, crowd scene, golden sunlight."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "此時的落雲宗，正舉辦天南修仙界百年盛事。韓立作為宗主，威震天南。",
    },
    {
        "id": 45,
        "title": "韓立登場 — 宗主之姿",
        "prompt": (
            "Han Li stands atop the ceremony platform in magnificent dark teal-green robes with "
            "gold embroidery, his long brown hair flowing in the wind. His expression is calm and "
            "absolute power radiates from him as golden spiritual pressure creates a visible aura. "
            "Thousands of cultivators below bow. Majestic ruler shot from low angle."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "元嬰後期巔峰——韓立，已是天南修仙界無可爭議的第一人。",
    },
    {
        "id": 46,
        "title": "韓立示威 — 擊敗兩大修士",
        "prompt": (
            "Epic battle scene: Han Li simultaneously fights two powerful cultivators in a private "
            "dimension. He deflects a wave of dark demonic energy with one hand and a beam of "
            "golden light with the other. His expression is serene despite the intense combat. "
            "Both opponents are forced back. Massive energy shockwaves, cinematic battle."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "一對二，韓立以碾壓之勢擊敗魏無涯與合歡老魔。天南再無敢與之爭鋒者。",
    },
    {
        "id": 47,
        "title": "韓立接到消息 — 表情驟變",
        "prompt": (
            "Han Li sits alone in a quiet study, reading a jade communication talisman. "
            "His calm expression suddenly shatters — eyes narrowing, jaw tightening, a flash of "
            "cold killing intent crosses his face. The talisman cracks in his grip. "
            "The serene atmosphere turns ice-cold. Dramatic emotional shift close-up."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "大典之後，一則消息傳來——宋玉、慕沛靈、柳玉三人，在坠魔谷中失聯。",
    },
    {
        "id": 48,
        "title": "韓立起身 — 殺意凜然",
        "prompt": (
            "Han Li stands abruptly, his robes billowing with unleashed spiritual pressure. "
            "Objects in the study float and rattle. His eyes glow with cold determination. "
            "Behind him, a ghostly green sword materializes from thin air. "
            "He walks toward the door with absolute purpose. "
            "Power unleashed, controlled fury, cinematic slow-motion."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "⋯⋯坠魔谷。是天煞魔尸。",
    },

    # ═══ ACT 7: 救援出發 ═══
    {
        "id": 49,
        "title": "韓立飛出落雲宗 — 極速",
        "prompt": (
            "Han Li launches from the sect peak like a meteor, leaving a brilliant green-gold "
            "light trail across the sky. His speed creates sonic boom-like shockwaves in the clouds. "
            "Cultivators below look up in awe at the streak of light. "
            "Extreme speed flight, dramatic vapor trail, golden hour lighting."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "韓立毫不猶豫，祭出風雷翅，以天南第一的速度直奔坠魔谷。",
    },
    {
        "id": 50,
        "title": "飛越山河 — 地圖視角",
        "prompt": (
            "Aerial tracking shot following a brilliant green light streak across a vast Chinese "
            "fantasy landscape — over jade-green rivers, past floating mountain islands, through "
            "cloud formations. The landscape rushes by below at impossible speed. "
            "Epic journey montage, beautiful xianxia world scenery."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "千里之路，在元嬰後期修士的全力飛行之下，不過片刻。",
    },
    {
        "id": 51,
        "title": "坠魔谷上空 — 韓立降臨",
        "prompt": (
            "Han Li hovers above the multicolored miasma of the Fall Devil Valley, looking down "
            "with ice-cold eyes. His teal-green robes flutter in the toxic wind. "
            "The miasma parts beneath his spiritual pressure like water before a ship's bow. "
            "He is a god descending upon a cursed land. Backlit by setting sun."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "我回來了⋯⋯坠魔谷。這一次，我要收回屬於我的一切。",
    },
    {
        "id": 52,
        "title": "韓立沖入谷中 — 瘴氣盡散",
        "prompt": (
            "Han Li plunges straight down into the valley. His golden spiritual aura expands "
            "massively, disintegrating the toxic miasma in a sphere around him. Ancient restriction "
            "arrays that once blocked passage shatter like glass before his power. "
            "He descends into the valley like a falling star. Overwhelming power display."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "曾經令無數修士葬身的瘴氣禁制，在元嬰後期的靈壓面前，不堪一擊。",
    },

    # ═══ ACT 8: 內谷深處 — 對峙 ═══
    {
        "id": 53,
        "title": "韓立進入內谷",
        "prompt": (
            "Han Li walks calmly through the inner valley, the surreal landscape bending to his "
            "presence. Bioluminescent plants dim as he passes. Spatial rifts that once threatened "
            "travelers stabilize in his spiritual pressure field. "
            "He follows traces of demonic qi like a hunter tracking prey. Purposeful stride."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "韓立循著玄陰魔氣的殘留，一步步走向內谷深處。",
    },
    {
        "id": 54,
        "title": "發現洞穴 — 禁制陣法",
        "prompt": (
            "Han Li stands before the cave sealed with dark red and black restriction arrays. "
            "He extends one hand, fingers splayed, and the demonic restriction glows intensely "
            "in response to his probing spiritual sense. His eyes narrow — recognition. "
            "The demonic array vs golden probing light, magical analysis scene."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "這禁制⋯⋯確實是我的路數。不過已經摻雜了天煞魔功的變化。",
    },
    {
        "id": 55,
        "title": "洞穴內 — 三女感應到韓立",
        "prompt": (
            "Inside the sealed cave, Song Yu's eyes snap open. She gasps — sensing an overwhelming "
            "familiar spiritual presence approaching from outside. Hope floods her face. "
            "Mu Peiling and Liu Yu also react, looking toward the sealed entrance. "
            "The restriction array on the door begins to flicker. Hope dawning on trapped faces."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "這靈壓⋯⋯是韓前輩！他來了！！",
    },
    {
        "id": 56,
        "title": "第二元嬰出現 — 韓立背後",
        "prompt": (
            "Behind Han Li, the air distorts and the Second Nascent Soul materializes from "
            "shadows. His skull face grins, fangs gleaming. His gold-silver arm crackles with "
            "dark energy. Swirling black demonic qi fills the valley around them. "
            "Two figures face off — original vs copy. Mirror confrontation."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "好久不見⋯⋯本體。你終於來了。",
    },
    {
        "id": 57,
        "title": "韓立轉身 — 面對自己的分身",
        "prompt": (
            "Han Li slowly turns to face the Second Nascent Soul. A long dramatic pause. "
            "Two figures stand in the inner valley — one in pristine teal-green robes radiating "
            "golden light, the other a dark demonic figure wreathed in black qi. "
            "Perfect mirror opposition. Epic face-off composition."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "⋯⋯你已經不是我了。但你欠我的，今天要一併算清。",
    },
    {
        "id": 58,
        "title": "對峙 — 氣場碰撞",
        "prompt": (
            "Spectacular VFX scene: golden spiritual pressure from Han Li and black demonic qi "
            "from the Second Nascent Soul collide in the space between them. The ground cracks, "
            "rocks float, and the sky distorts. A shockwave spreads outward. "
            "Neither gives ground. Pure power confrontation, Dragon Ball-like energy clash."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "兩股截然不同的力量在內谷碰撞。本體元嬰，對峙天煞魔尸——一場宿命之戰，一觸即發。",
    },

    # ═══ ACT 9: 激戰 ═══
    {
        "id": 59,
        "title": "第二元嬰先出手 — 黑焰攻擊",
        "prompt": (
            "The Second Nascent Soul attacks first, launching a torrent of black demonic fire "
            "from his metallic arm. The black flames take the shape of screaming demonic skulls "
            "as they rush toward Han Li. The inner valley ignites with dark fire. "
            "Intense combat VFX, black fire wave attack."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "受死吧！！",
    },
    {
        "id": 60,
        "title": "韓立以劍破焰",
        "prompt": (
            "Han Li draws his jade-green flying sword. With a single calm slash, a crescent of "
            "blinding green-gold sword qi bisects the black flame attack perfectly. "
            "The divided flames dissipate harmlessly to either side. Han Li stands unmoved. "
            "Effortless defense, sword light vs dark fire, elegant power."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "就這？",
    },
    {
        "id": 61,
        "title": "激烈交鋒 — 近身戰",
        "prompt": (
            "Fast-paced close combat between Han Li and the Second Nascent Soul. Green sword light "
            "clashes against the gold-silver metallic arm. Shockwaves from each blow. "
            "They move at incredible speed between the floating rocks of the inner valley. "
            "Dynamic camera angles, speed lines, each strike creating light explosions."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "兩人在內谷中展開了激烈的交鋒，每一招都足以摧毀一座山峰。",
    },
    {
        "id": 62,
        "title": "第二元嬰使出殺招 — 天煞魔功",
        "prompt": (
            "The Second Nascent Soul roars, his body expanding as the Heavenly Fiend Corpse form "
            "fully activates. He grows larger, more demonic, black armor-like scales forming on "
            "his body. A massive dark aura in the shape of a demonic god appears behind him. "
            "Ultimate technique activation, dark transformation, terrifying power-up."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "天煞魔功——第七層！！讓你看看，我在這谷中修煉多年的成果！",
    },
    {
        "id": 63,
        "title": "韓立冷笑 — 真正的力量",
        "prompt": (
            "Han Li smirks coldly. His golden spiritual pressure intensifies dramatically — "
            "the entire inner valley is bathed in golden light. Multiple flying swords materialize "
            "around him forming a complex sword formation. His Wind Thunder Wings (electric blue "
            "and gold feathered wings) spread from his back. Ascendant power reveal."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "你修煉了幾年？我已是元嬰後期巔峰。差距⋯⋯你自己清楚。",
    },
    {
        "id": 64,
        "title": "韓立終極一擊 — 萬劍歸宗",
        "prompt": (
            "Han Li raises one hand toward the sky. Hundreds of golden-green flying swords "
            "materialize in the air above, forming a massive sword formation that blots out "
            "the distorted sky. They all align and plunge downward in unison toward the "
            "Second Nascent Soul. A pillar of golden sword light descends like divine judgment. "
            "Ultimate attack, godlike power, overwhelming beautiful destruction."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "萬劍歸宗——韓立以天南第一人的全力，向他曾經的分身發動了最終一擊。",
    },
    {
        "id": 65,
        "title": "第二元嬰被壓制",
        "prompt": (
            "The Second Nascent Soul is overwhelmed by the golden sword rain. His dark aura "
            "shatters layer by layer. The demonic qi dissipates under the golden onslaught. "
            "He falls to his knees, cracks spreading across his corpse body. "
            "Defeat scene, dark power crumbling before light, dramatic lighting."
        ),
        "voice": VOICE_SECOND_SOUL,
        "dialogue": "不⋯⋯不可能⋯⋯！",
    },
    {
        "id": 66,
        "title": "收回第二元嬰",
        "prompt": (
            "Han Li stands over the defeated Second Nascent Soul. He extends his palm, and a "
            "golden suction force pulls a small glowing dark infant-like spirit (the nascent soul) "
            "out of the corpse body. The spirit struggles but is drawn into a jade bottle. "
            "The Heavenly Fiend Corpse collapses. Reclamation, golden light absorbing darkness."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "⋯⋯回來吧。你本就是我的一部分。",
    },

    # ═══ ACT 10: 救出 — 結尾 ═══
    {
        "id": 67,
        "title": "韓立破開洞穴封印",
        "prompt": (
            "Han Li places his palm on the sealed cave entrance. The dark restriction array "
            "glows red, resisting, then shatters with a brilliant flash as Han Li's golden "
            "spiritual power overwhelms it. The cave door crumbles, golden light floods inside. "
            "Liberation scene, light breaking through darkness."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "韓立一掌打碎洞穴封印，金色靈光傾瀉而入。",
    },
    {
        "id": 68,
        "title": "三女得救 — 宋玉抬頭",
        "prompt": (
            "Inside the cave, the restriction chains binding the three women dissolve in golden "
            "light. Song Yu lifts her head, her clear bright eyes meeting Han Li's gaze as he "
            "stands silhouetted in the cave entrance, golden aura glowing behind him like a halo. "
            "Tears of relief form in her eyes. Beautiful rescue moment, emotional release."
        ),
        "voice": VOICE_SONGYU,
        "dialogue": "韓前輩⋯⋯我就知道⋯⋯你一定會來。",
    },
    {
        "id": 69,
        "title": "韓立幫宋玉解除禁制",
        "prompt": (
            "Han Li gently places his hand near Song Yu's shoulder (not touching), sending soft "
            "golden healing energy to dissolve the remaining restrictions. She stumbles as the "
            "bindings release; he steadies her with one hand. A brief moment of closeness. "
            "Tender rescue scene, warm golden light, gentle interaction."
        ),
        "voice": VOICE_HANLI,
        "dialogue": "傷到了嗎？",
    },
    {
        "id": 70,
        "title": "慕沛靈和柳玉獲救",
        "prompt": (
            "Han Li releases the other two women from their bindings. Mu Peiling collapses in "
            "grateful relief, clutching her jade pendant. Liu Yu bows deeply despite her weakness. "
            "All three women are freed but weakened. Warm rescue aftermath."
        ),
        "voice": VOICE_MUPEILING,
        "dialogue": "謝謝⋯⋯韓前輩⋯⋯謝謝你。",
    },
    {
        "id": 71,
        "title": "走出洞穴 — 重見天日",
        "prompt": (
            "The four figures walk out of the dark cave into the inner valley. The surreal landscape "
            "is bathed in warm amber light. Song Yu walks beside Han Li, the other two follow behind. "
            "After the darkness of captivity, the open sky feels vast and beautiful. "
            "Emergence from darkness into light, rebirth symbolism."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "走出黑暗，重見天日。坠魔谷的危機，終於化解。",
    },
    {
        "id": 72,
        "title": "宋玉側臉 — 偷看韓立",
        "prompt": (
            "Close-up of Song Yu's side profile as she walks. She steals a glance at Han Li "
            "walking beside her — gratitude, admiration, and a hint of deeper emotion in her "
            "extraordinarily clear eyes. A small, warm, genuine smile forms on her lips. "
            "She quickly looks away when he might notice. Subtle romantic moment."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "宋玉悄悄看了韓立一眼，嘴角不禁微微上揚。",
    },
    {
        "id": 73,
        "title": "飛出坠魔谷 — 四人同行",
        "prompt": (
            "Four figures fly out of the valley on a massive golden sword platform created by "
            "Han Li. The toxic miasma parts before them. They rise above the dark mountain range "
            "into a spectacular sunset sky. The cursed valley shrinks below them. "
            "Freedom flight, golden hour, beautiful clouds."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "四人飛出坠魔谷，瘴氣在身後漸漸遠去。一場劫難，終成往事。",
    },
    {
        "id": 74,
        "title": "遠處落雲宗 — 回家",
        "prompt": (
            "Wide shot from behind: four silhouetted figures on the golden sword platform flying "
            "toward the distant Luoyun Sect floating mountains, now glowing golden in the sunset. "
            "Clouds part to reveal the beautiful sect. A sense of home, safety, and belonging. "
            "Warm golden tones, epic homecoming shot."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "前方，落雲宗的峰巒在夕陽中若隱若現。回家的路，從不曾如此溫暖。",
    },
    {
        "id": 75,
        "title": "片尾 — 標題卡",
        "prompt": (
            "Black background slowly reveals elegant Chinese calligraphy text in gold: "
            "'凡人修仙傳 — 坠魔谷篇' and below it '宋玉之劫'. Subtle particles of golden light "
            "drift upward like fireflies. The text glows with warm inner light. "
            "Clean title card, elegant, cinematic end card."
        ),
        "voice": VOICE_NARRATOR,
        "dialogue": "凡人修仙傳——坠魔谷篇。",
    },
]


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


async def generate_tts(scene, output_dir):
    """Generate TTS audio for a scene using edge-tts."""
    import edge_tts

    audio_file = output_dir / f"voice_{scene['id']:02d}.mp3"
    if audio_file.exists():
        return audio_file

    voice = scene.get("voice", VOICE_NARRATOR)
    dialogue = scene.get("dialogue", "")
    if not dialogue:
        return None

    # Differentiate characters by rate/pitch
    rate = "-5%"
    if voice == VOICE_SECOND_SOUL:
        rate = "-15%"  # Slower, more menacing
    elif voice == VOICE_LIUYU:
        rate = "+5%"   # Slightly faster, more assertive
    elif voice == VOICE_SONGYU:
        rate = "-8%"   # Gentle, measured
    communicate = edge_tts.Communicate(dialogue, voice, rate=rate)
    await communicate.save(str(audio_file))
    return audio_file


def generate_scene_video(client, scene, progress, max_retries=5):
    """Generate video clip for a scene using Veo 3.1 with retry on rate limit."""
    scene_id = str(scene["id"])
    output_file = OUTPUT_DIR / f"scene_{scene['id']:02d}.mp4"

    # Skip if already generated
    if scene_id in progress and progress[scene_id].get("status") == "done" and output_file.exists():
        print(f"  ✓ Scene {scene['id']} already generated, skipping")
        return output_file

    full_prompt = STYLE + scene["prompt"]

    print(f"\n{'='*60}")
    print(f"  Scene {scene['id']}: {scene['title']}")
    print(f"{'='*60}")

    for attempt in range(max_retries):
        print(f"  Generating {VIDEO_DURATION}s video at {RESOLUTION}... (attempt {attempt+1}/{max_retries})")

        try:
            operation = client.models.generate_videos(
                model=MODEL,
                prompt=full_prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio=ASPECT_RATIO,
                    duration_seconds=VIDEO_DURATION,
                ),
            )

            # Poll until done
            poll_count = 0
            while not operation.done:
                poll_count += 1
                elapsed = poll_count * 15
                print(f"  Waiting... ({elapsed}s elapsed)", end="\r")
                time.sleep(15)
                operation = client.operations.get(operation)

            print(f"  Generation complete!                    ")

            # Check if we got valid video
            if not operation.response or not operation.response.generated_videos:
                print(f"  WARNING: No video returned, retrying...")
                time.sleep(30)
                continue

            # Download and save
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(str(output_file))

            print(f"  Saved: {output_file}")

            # Update progress
            progress[scene_id] = {
                "title": scene["title"],
                "file": str(output_file),
                "status": "done",
            }
            save_progress(progress)

            return output_file

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 60 * (2 ** attempt)  # 60s, 120s, 240s, 480s, 960s
                print(f"  Rate limited! Waiting {wait_time}s before retry... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                print(f"  ERROR generating scene {scene['id']}: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in 30s...")
                    time.sleep(30)
                    continue
                progress[scene_id] = {"title": scene["title"], "status": f"error: {e}"}
                save_progress(progress)
                return None

    print(f"  FAILED after {max_retries} attempts for scene {scene['id']}")
    progress[scene_id] = {"title": scene["title"], "status": "error: max retries exceeded"}
    save_progress(progress)
    return None


def merge_audio_video(video_file, audio_file, output_file):
    """Merge TTS audio onto video using ffmpeg, with audio padding/trimming to match video duration."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_file),
        "-i", str(audio_file),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # If audio is longer than video, pad video
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-i", str(audio_file),
            "-filter_complex",
            f"[1:a]apad=pad_dur=0[a];[0:v]setpts=PTS-STARTPTS[v]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_file),
        ]
        subprocess.run(cmd, capture_output=True, text=True)
    return output_file


def add_ambient_sound(video_file, output_file, scene):
    """Add ambient xianxia sound effects based on scene content."""
    # Use ffmpeg to add subtle reverb/atmospheric effect to existing audio
    # This simulates the donghua atmosphere without needing external SFX files
    prompt_lower = scene["prompt"].lower()

    # Determine ambient filter based on scene type
    if any(w in prompt_lower for w in ["battle", "combat", "attack", "fight", "clash", "sword"]):
        # Battle scenes: add bass boost and echo
        audio_filter = "aecho=0.8:0.7:40:0.5,equalizer=f=80:t=h:w=200:g=3"
    elif any(w in prompt_lower for w in ["cave", "dark", "sealed", "prison"]):
        # Cave scenes: add reverb
        audio_filter = "aecho=0.8:0.9:500:0.3"
    elif any(w in prompt_lower for w in ["fly", "speed", "wind", "aerial"]):
        # Flight scenes: add wind-like filter
        audio_filter = "highpass=f=200,aecho=0.6:0.3:20:0.4"
    else:
        # Default: subtle ambient
        audio_filter = "aecho=0.8:0.5:100:0.3"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_file),
        "-af", audio_filter,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # If filter fails, just copy
        subprocess.run(["cp", str(video_file), str(output_file)])
    return output_file


def concatenate_videos(clip_files):
    """Use ffmpeg to concatenate all scene clips into one video."""
    concat_list = OUTPUT_DIR / "concat_list.txt"
    with open(concat_list, "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    print(f"\nConcatenating {len(clip_files)} clips into final video...")

    # Re-encode for consistency (different scenes may have different codecs)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        str(FINAL_OUTPUT),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Final video saved: {FINAL_OUTPUT}")
    else:
        print(f"  FFmpeg error: {result.stderr[:500]}")


async def generate_all_tts(scenes):
    """Generate all TTS audio files."""
    tts_dir = OUTPUT_DIR / "tts"
    tts_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Generating TTS for {len(scenes)} scenes...")
    print(f"{'#'*60}")

    results = {}
    for scene in scenes:
        audio = await generate_tts(scene, tts_dir)
        if audio:
            results[scene["id"]] = audio
            print(f"  ✓ Scene {scene['id']}: {scene['title']}")
        else:
            print(f"  - Scene {scene['id']}: no dialogue")

    print(f"  TTS complete: {len(results)} audio files")
    return results


def main():
    parser = argparse.ArgumentParser(description="凡人修仙傳 坠魔谷篇 Video Generator")
    parser.add_argument("--scene", type=int, help="Generate only this scene number")
    parser.add_argument("--concat", action="store_true", help="Only concatenate existing clips")
    parser.add_argument("--list", action="store_true", help="List all scenes")
    parser.add_argument("--tts-only", action="store_true", help="Only generate TTS audio")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS generation")
    parser.add_argument("--duration", type=int, default=8, choices=[4, 6, 8],
                        help="Video duration per scene (default: 8)")
    args = parser.parse_args()

    global VIDEO_DURATION
    VIDEO_DURATION = args.duration

    # List scenes
    if args.list:
        print("\n凡人修仙傳 坠魔谷篇 — 場景列表\n")
        acts = {
            1: "ACT 1: 落雲宗 — 出發前",
            7: "ACT 2: 坠魔谷外圍",
            13: "ACT 3: 進入內谷",
            21: "ACT 4: 追殺 — 黑雲降臨",
            31: "ACT 5: 洞穴囚禁 — 真相揭曉",
            44: "ACT 6: 韓立震天南",
            49: "ACT 7: 救援出發",
            53: "ACT 8: 內谷深處 — 對峙",
            59: "ACT 9: 激戰",
            67: "ACT 10: 救出 — 結尾",
        }
        for s in SCENES:
            if s["id"] in acts:
                print(f"\n  ═══ {acts[s['id']]} ═══")
            print(f"  Scene {s['id']:2d}: {s['title']}")
            if s.get("dialogue"):
                print(f"           「{s['dialogue'][:40]}{'...' if len(s.get('dialogue','')) > 40 else ''}」")
        total_secs = len(SCENES) * VIDEO_DURATION
        print(f"\n  Total: {len(SCENES)} scenes × {VIDEO_DURATION}s = {total_secs}s ({total_secs//60}m{total_secs%60}s)")
        return

    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)

    # TTS only
    if args.tts_only:
        asyncio.run(generate_all_tts(SCENES))
        return

    # Concat only
    if args.concat:
        clip_files = sorted(OUTPUT_DIR.glob("final_*.mp4"))
        if not clip_files:
            clip_files = sorted(OUTPUT_DIR.glob("scene_*.mp4"))
        if not clip_files:
            print("No clips found to concatenate!")
            return
        concatenate_videos(clip_files)
        return

    # Initialize client
    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    progress = load_progress()

    # Filter scenes
    if args.scene:
        scenes_to_gen = [s for s in SCENES if s["id"] == args.scene]
        if not scenes_to_gen:
            print(f"Scene {args.scene} not found (valid: 1-{len(SCENES)})")
            return
    else:
        scenes_to_gen = SCENES

    print(f"\n{'#'*60}")
    print(f"  凡人修仙傳 — 坠魔谷篇（宋玉之劫）")
    print(f"  Scenes: {len(scenes_to_gen)} | Duration: {VIDEO_DURATION}s each")
    print(f"  Model: {MODEL} | Resolution: {RESOLUTION}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'#'*60}")

    # Step 1: Generate TTS for all scenes
    if not args.no_tts:
        tts_results = asyncio.run(generate_all_tts(scenes_to_gen))
    else:
        tts_results = {}

    # Step 2: Generate video for each scene
    generated_files = []
    for scene in scenes_to_gen:
        video_file = generate_scene_video(client, scene, progress)

        if video_file:
            # Step 3: Merge TTS audio with video
            if scene["id"] in tts_results:
                merged_file = OUTPUT_DIR / f"merged_{scene['id']:02d}.mp4"
                print(f"  Merging audio for scene {scene['id']}...")
                merge_audio_video(video_file, tts_results[scene["id"]], merged_file)

                # Step 4: Add ambient sound effects
                final_file = OUTPUT_DIR / f"final_{scene['id']:02d}.mp4"
                add_ambient_sound(merged_file, final_file, scene)
                generated_files.append(final_file)
            else:
                # No dialogue, just add ambient
                final_file = OUTPUT_DIR / f"final_{scene['id']:02d}.mp4"
                add_ambient_sound(video_file, final_file, scene)
                generated_files.append(final_file)

        # Rate limiting
        if scene != scenes_to_gen[-1]:
            print("  Cooling down 5s before next scene...")
            time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Generation Summary")
    print(f"{'='*60}")
    success = len(generated_files)
    total = len(scenes_to_gen)
    print(f"  Success: {success}/{total}")

    if success < total:
        failed = [s["id"] for s in scenes_to_gen
                  if not (OUTPUT_DIR / f"final_{s['id']:02d}.mp4").exists()]
        print(f"  Failed: {failed}")
        print(f"  Re-run with --scene N to retry individual scenes")

    # Concatenate if all scenes were generated
    if not args.scene:
        all_clips = sorted(OUTPUT_DIR.glob("final_*.mp4"))
        if len(all_clips) == len(SCENES):
            concatenate_videos(all_clips)
        else:
            print(f"\n  {len(all_clips)}/{len(SCENES)} clips ready. "
                  f"Run with --concat after all scenes are generated.")


if __name__ == "__main__":
    main()
