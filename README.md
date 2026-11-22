# SilentSpeak AI: Real-Time Silent Speech Recognition

A multimodal assistive system that converts silent lip movements and facial muscle activity into synthetic speech, helping people who cannot vocalize to communicate.

## Overview

Patients who cannot speak, for example in hospital or home-care settings, often struggle to communicate basic needs. SilentSpeak AI combines **computer vision** with **surface electromyography (EMG)** from the jaw and cheek to recognize what a person is trying to say without any sound, then speaks it aloud using text-to-speech.

## Key Results

- **93% test accuracy** on a 20-word medical and daily-needs vocabulary
- Robust to **low light** and **mouth occlusion**, since the EMG channel keeps working when the camera view degrades

## Features

- **Multimodal fusion** of visual lip features and surface EMG signals
- **480 facial landmark features extracted per frame**
- **Deep learning classifier** over a 20-word vocabulary
- **Short user calibration** to adapt to each person
- **Instant text-to-speech** output
- Designed for **hospital, home-care and assistive-living** use

## System Architecture

1. **Vision branch:** camera frames → facial landmark extraction (480 features per frame)
2. **EMG branch:** surface electrodes on the jaw and cheek → signal processing
3. **Fusion model:** multimodal deep learning classifier combines both streams
4. **Output:** predicted word → text → synthetic speech

## How It Works

1. The user completes a short calibration.
2. The camera captures lip and facial movement while EMG sensors record jaw and cheek muscle activity.
3. Facial landmarks and EMG features are extracted and fused.
4. The classifier predicts the intended word from the 20-word vocabulary.
5. The word is displayed as text and spoken aloud immediately.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 93% |
| Vocabulary size | 20 words |
| Input modalities | Computer vision + surface EMG |
| Robustness | Low light, mouth occlusion |

## Tech Stack

- **Languages:** Python
- **Computer vision:** facial landmark detection
- **Machine learning:** multimodal deep learning
- **Signal acquisition:** surface EMG (jaw and cheek)
- **Output:** text-to-speech

## My Role

Solo project built at HowNwhy Education. I designed the sensing setup, built the feature-extraction pipeline, trained the multimodal classifier, and implemented the calibration and text-to-speech modules.

## Future Work

- Larger vocabulary and continuous, sentence-level recognition
- Reduced calibration time and better multi-user generalization
- Embedded or on-device deployment

## Disclaimer

This is a research and educational prototype, **not a certified medical device**.

## Author

**Akash Krishna**
ECE Graduate | Embedded Systems | Edge AI
[LinkedIn](https://linkedin.com/in/akash-krishna-729685343) | akashkrishna221@gmail.com
