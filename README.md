# 👁️ Eye Blink Morse Code Detector

## _Hands-Free Communication Through Computer Vision_

> 🚀 **Transform your eye blinks into text** using advanced computer vision and machine learning. This innovative accessibility tool converts intentional eye blinks into Morse code, enabling hands-free text input with intelligent differentiation between natural and code blinks.

## 🎥 **Demo Video**

[📺 **Watch the Eye Blink Morse Code Demo**](eyeblinkmorse.mp4)

> 👆 **See it in action!** Watch how eye blinks are converted to Morse code in real-time with our intelligent dual-mode system.

---

## ✨ What Makes This Special

🎯 **Smart Code Mode** • Double-blink activation separates natural from intentional blinks  
🧠 **AI-Powered Detection** • 468-point facial landmark tracking with MediaPipe  
⚡ **Intelligent Blink Analysis** • Advanced EAR algorithm with customizable thresholds  
🔄 **Backspace Support** • Eight-dot sequence for error correction  
📊 **Live Visual Feedback** • Real-time overlay with mode status and decoded text  
🎵 **Morse Magic** • Intelligent timing classification for dots and dashes

---

## 🎬 The Experience

**Code Mode System:** Normal blinking is ignored until you double-blink quickly to activate code mode. Then short blinks become **dots** (•), long blinks become **dashes** (–), and intelligent timing separates letters and words. Built with accessibility in mind and perfect for learning computer vision concepts.

### 🔧 Core Technologies

```
OpenCV ──────► Real-time video processing & display
MediaPipe ───► 468-point facial mesh detection
NumPy ───────► High-performance mathematical operations
```

### 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    👁️ Eye Blink Morse System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📹 Webcam Input                                               │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   OpenCV     │───►│  MediaPipe   │───►│   EAR Calc   │     │
│  │ Video Capture│    │ Face Mesh    │    │ Eye Analysis │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                 │              │
│                                                 ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Text Output  │◄───│ Morse Decode │◄───│ Blink Filter │     │
│  │ & File Save  │    │ & Commands   │    │ & Timing     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
│  🔄 Dual Mode System: Normal ⇄ Code                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 📦 Installation

**Step 1:** Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

**Step 2:** Run the application

```bash
python eye_blink_morse.py
```

### � **Usage Flow Diagram**

```
                    👁️ Eye Blink Morse Code - Usage Flow
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  🚀 Start Application                                                   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────┐    ⚙️ Press 'C'    ┌─────────────┐                    │
│  │ 🟢 Normal   │──────────────────►│ 🔧 Calibrate │                    │
│  │ Mode        │                   │ Threshold   │                    │
│  │ (Default)   │◄──────────────────│ Mode        │                    │
│  └─────────────┘    Ready to Use   └─────────────┘                    │
│       │                                                                 │
│       │ 👀 Double Blink or Press 'M'                                   │
│       ▼                                                                 │
│  ┌─────────────┐                                                       │
│  │ 🔴 Code     │                                                       │
│  │ Mode        │ ─────► 👁️ Short Blink = • (DOT)                     │
│  │ (Active)    │ ─────► 👁️ Long Blink = ─ (DASH)                     │
│  │             │ ─────► ⏱️ 1.5s pause = Letter commit                 │
│  │             │ ─────► ⏱️ 7.0s pause = Word space                   │
│  └─────────────┘                                                       │
│       │                                                                 │
│       │ ⏱️ 10s timeout or Manual toggle                               │
│       ▼                                                                 │
│  ┌─────────────┐    📝 Press 'S'    ┌─────────────┐                    │
│  │ 🟢 Normal   │──────────────────►│ 💾 Save     │                    │
│  │ Mode        │                   │ Message     │                    │
│  │ (Return)    │                   │ to File     │                    │
│  └─────────────┘                   └─────────────┘                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### �🎮 Controls & Features

#### 🔄 **Code Mode Activation**

- **Double-blink quickly** (within 0.5s) to enter code mode
- **Manual toggle:** Press `M` key to activate/deactivate
- **Auto-timeout:** Code mode deactivates after 10 seconds of inactivity

#### ⌨️ **Keyboard Controls**

| Key | Action          | Description                                   |
| --- | --------------- | --------------------------------------------- |
| `Q` | **Quit**        | Exit the application                          |
| `R` | **Reset**       | Clear current message and sequence            |
| `P` | **Pause**       | Toggle detection pause                        |
| `M` | **Mode Toggle** | Manually activate/deactivate code mode        |
| `H` | **Help**        | Show detailed help information                |
| `C` | **Calibrate**   | Enter EAR threshold calibration mode          |
| `S` | **Save**        | Save current message to `morse_output.txt`    |
| `T` | **Auto-Adjust** | Auto-calibrate threshold based on current EAR |
| `↑` | **Increase**    | Increase EAR threshold                        |
| `↓` | **Decrease**    | Decrease EAR threshold                        |

---

## 🧠 How It Works

### 🔍 **Dual-Mode System**

```
┌─────────────────┐    Double Blink    ┌─────────────────┐
│   🟢 Normal      │ ─────────────────► │   🔴 Code       │
│   Mode          │                    │   Mode          │
│                 │                    │                 │
│ • Natural blinks│                    │ • Morse input   │
│ • Ignored       │ ◄───── Timeout ────│ • Timed blinks  │
│ • No decoding   │    (10 seconds)    │ • Text output   │
└─────────────────┘                    └─────────────────┘
```

1. **🟢 Normal Mode** - All blinks are ignored, allowing natural blinking
2. **🔴 Code Mode** - Blinks are interpreted as Morse code input

### 👁️ **Eye Aspect Ratio (EAR) Detection**

```
┌─────────────────────────────────────────────────────┐
│                Eye Landmark Points                  │
│                                                     │
│    p1 ●────────────────────────● p4                │
│       │                        │                   │
│       │     p2 ●──────● p6     │                   │
│       │        │      │        │                   │
│       │     p3 ●──────● p5     │                   │
│       │                        │                   │
│    ● EAR Calculation:          │                   │
│    (||p₂-p₆|| + ||p₃-p₅||) / (2||p₁-p₄||)         │
└─────────────────────────────────────────────────────┘

📊 EAR Values:
┌─────────┬─────────┬────────────┐
│  State  │   EAR   │   Status   │
├─────────┼─────────┼────────────┤
│  Open   │  0.25+  │ 🟢 Normal  │
│  Blink  │  0.15-  │ 🔴 Detect  │
│ Closed  │  0.10-  │ 🔴 Trigger │
└─────────┴─────────┴────────────┘
```

**The Science:** EAR values drop significantly during blinks, creating a reliable detection signal that's calibrated to your specific eye shape and lighting conditions.

### 🎯 **Detection Pipeline**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 📹 Video    │───►│ 🎯 Face     │───►│ 👁️ Eye      │───►│ 📐 EAR      │
│ Capture     │    │ Detection   │    │ Landmarks   │    │ Calculate   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                    │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│ ✍️ Morse    │◄───│ ⏱️ Pattern   │◄───│ 🔄 Blink    │◄─────────────┘
│ Translation │    │ Analysis    │    │ Detection   │
└─────────────┘    └─────────────┘    └─────────────┘

📊 Processing Flow:
1. 📹 Capture → Real-time webcam video processing
2. 🎯 Detect → MediaPipe extracts facial landmarks
3. 📐 Calculate → Compute smoothed EAR values
4. 🔄 Classify → Determine natural vs. intentional blinks
5. ⏱️ Decode → Convert timing to dots/dashes
6. ✍️ Process → Translate Morse to text with special commands
```

---

## 📝 Morse Code Features

### 🔤 **Character Support**

- **Letters:** A-Z (complete alphabet)
- **Numbers:** 0-9 (full numeric range)
- **Punctuation:** `.`, `,`, `?`, `'`, `!`, `/`, `(`, `)`, `&`, `:`, `;`, `=`, `+`, `-`, `_`, `"`, `$`, `@`

### � **Morse Code Reference Chart**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Common Letters & Numbers                     │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┤
│ A  •─   │ B ─••• │ C ─•─•  │ D ─••   │ E •     │ F ••─•  │...  │
│ G ──•   │ H ••••  │ I ••    │ J •───  │ K ─•─   │ L •─••  │...  │
│ M ──    │ N ─•    │ O ───   │ P •──•  │ Q ──•─  │ R •─•   │...  │
│ S •••   │ T ─     │ U ••─   │ V •••─  │ W •──   │ X ─••─  │...  │
│ Y ─•──  │ Z ──••  │         │         │         │         │     │
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────┤
│ 0 ───── │ 1 •──── │ 2 ••─── │ 3 •••─- │ 4 ••••─ │ 5 ••••• │     │
│ 6 ─•••• │ 7 ──••• │ 8 ───•• │ 9 ────• │         │         │     │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────┘

Quick Reference: • = Short Blink, ─ = Long Blink
SOS Example: ••• ─── ••• (3 shorts, 3 longs, 3 shorts)
```

### �🛠️ **Special Commands**

| Pattern    | Command       | Action                         |
| ---------- | ------------- | ------------------------------ |
| `........` | **BACKSPACE** | Delete last character (8 dots) |

### ⏱️ **Timing Logic**

```
Blink Duration Analysis:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Short Blink (< 0.3s)           Long Blink (≥ 0.3s)   │
│  ┌─────┐                        ┌─────────────┐         │
│  │ ●   │ = DOT                  │ ●───────────│ = DASH  │
│  └─────┘                        └─────────────┘         │
│                                                         │
└─────────────────────────────────────────────────────────┘

Timing Gaps:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Letter Gap (1.5s):    • • •   ─   [PAUSE]  ┌─Letter─┐  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────┐         │   A    │  │
│  │ ●   │ │ ●   │ │ ●   │ │ ●───────│         └────────┘  │
│  └─────┘ └─────┘ └─────┘ └─────────┘                     │
│                                                          │
│  Word Gap (7.0s):      [LONG PAUSE]         ┌─Space─┐   │
│  ┌─────────────────────────────────────────► │   _   │   │
│  │           7 seconds silence               └───────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

- **Short blinks** (< 0.3s) = Dots (•)
- **Long blinks** (≥ 0.3s) = Dashes (–)
- **Letter gap:** 1.5s silence commits current pattern
- **Word gap:** 7.0s silence adds automatic space

---

## ⚙️ Configuration Parameters

| Setting           | Default | Purpose                            |
| ----------------- | ------- | ---------------------------------- |
| **EAR Threshold** | `0.15`  | Blink detection sensitivity        |
| **Dot/Dash Time** | `0.3s`  | Duration cutoff for dots vs dashes |
| **Letter Gap**    | `1.5s`  | Time to commit a Morse letter      |
| **Word Gap**      | `7.0s`  | Time to add automatic space        |
| **Code Timeout**  | `10.0s` | Auto-deactivate code mode          |
| **Debounce Time** | `0.08s` | Minimum time between blinks        |

---

## 🎯 Eye Landmark Mapping

### 👁️ **Selected Landmarks for EAR Calculation**

**Right Eye:** `[33, 160, 158, 133, 153, 144]`  
**Left Eye:** `[362, 385, 387, 263, 373, 380]`

> 📍 **Optimized Selection:** These 6 points per eye provide stable EAR calculation while maintaining real-time performance.

---

## 🔧 Calibration Guide

### 🎛️ **Finding Your Perfect Threshold**

```
Calibration Flow:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Press 'C' │───►│ Open Eyes   │───►│ Press ↑ if │───►│ Close Eyes  │
│ Enter Calib │    │ Keep Steady │    │ Shows CLOSED│    │ Keep Steady │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                            │                     │
┌─────────────┐    ┌─────────────┐         │                     │
│ Test Double │◄───│ Press 'T'   │◄────────┘                     │
│ Blink Detect│    │ Auto-Adjust │                               │
└─────────────┘    └─────────────┘                               │
                         ▲                                       │
                         │                                       ▼
                   ┌─────────────┐    ┌─────────────────────────────┐
                   │ Perfect!    │◄───│ Press ↓ if Shows OPEN      │
                   │ Ready to Go │    │ (When Eyes Closed)         │
                   └─────────────┘    └─────────────────────────────┘

📊 Visual Feedback:
┌──────────────┬─────────────┬─────────────────┐
│   Eye State  │    Color    │   EAR Reading   │
├──────────────┼─────────────┼─────────────────┤
│ 👁️ OPEN      │ 🟢 Green    │ 0.20 - 0.30     │
│ 😑 CLOSED    │ 🔴 Red      │ 0.10 - 0.15     │
│ ⚙️ THRESHOLD │ 🔵 Blue     │ 0.15 (default)  │
└──────────────┴─────────────┴─────────────────┘
```

1. **Press `C`** to enter calibration mode
2. **Keep eyes open** and press `↑` to increase threshold if showing "CLOSED"
3. **Close eyes** and press `↓` to decrease threshold if showing "OPEN"
4. **Press `T`** for automatic calibration based on current EAR
5. **Test activation** with double-blinks when calibrated### 🎯 **Visual Feedback**

- **🟢 Green eye state** = Eyes detected as OPEN
- **🔴 Red eye state** = Eyes detected as CLOSED
- **EAR values** displayed in real-time for fine-tuning
- **Mode indicators** show current system state clearly

---

## � Pro Tips for Best Results

### 🌟 **Optimal Setup**

- **💡 Even lighting** on your face works best
- **📱 Stable positioning** about arm's length from camera
- **🎯 Deliberate blinks** - make them intentional and distinct
- **⏰ Consistent timing** - practice your dot/dash rhythm

### 🎭 **Mastering Code Mode**

- **Practice double-blink activation** until it becomes natural
- **Wait for visual confirmation** of code mode before starting
- **Use the backspace** (8 dots) to correct mistakes
- **Let timeouts work** - pauses automatically separate letters and words

---

## 🚀 Advanced Features

### � **Message Persistence**

- **Auto-save:** Press `S` to save messages to `morse_output.txt`
- **Timestamps:** Each saved message includes date/time
- **Append mode:** Messages accumulate in the file

### 🎨 **Visual Interface**

- **Real-time mode display** with color coding
- **Progress indicators** for special command patterns
- **Eye state visualization** with individual EAR values
- **Countdown timers** for code mode timeout

---

## 🎓 Educational Value

Perfect for learning:

- **Computer Vision** fundamentals and real-time processing
- **Signal Processing** with smoothing and hysteresis
- **Accessibility Technology** design principles
- **Human-Computer Interaction** innovation
- **Python** multimedia and OpenCV applications
- **MediaPipe** facial landmark detection

---

## 🌟 Built for Accessibility

This project demonstrates how computer vision can create **inclusive technology**. Whether for assistive communication, hands-free interaction, or educational exploration, it showcases the power of combining **artificial intelligence** with **human creativity**.

---

_💫 "Every intentional blink is a keystroke. Every sequence tells a story."_
