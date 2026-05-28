<p align="center">
  <img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width="300" alt="Python Logo">
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/your-username/your-repo"><img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-lightgrey.svg" alt="Platform"></a>
</p>

---

## 🖐️ Finger Draw

An AI-powered air drawing app — draw in the air using just your hand and a webcam.

---

## 📋 Requirements

- Python **3.11+**
- Webcam
- `opencv-python`
- `mediapipe`
- `numpy`

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/finger-draw.git
cd finger-draw
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe==0.10.14 numpy
```

### 4. Run the app

```bash
python finger_draw.py
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `C` | Clear canvas |
| `Q` / `Esc` | Quit |

---

## ✨ Features

- 👆 Index finger — draw
- 👍 Thumb — erase
- 🤏 Two hands pinch — zoom in / out
- ✊ Fist → open hand — switch color
- 📐 Auto straight line detection
- ⭕ Auto circle detection
- 😶 Real-time face blur

---

## 📄 License

MIT License
