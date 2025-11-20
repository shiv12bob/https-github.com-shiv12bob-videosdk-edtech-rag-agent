#!/usr/bin/env python3
import argparse, os
from dotenv import load_dotenv
load_dotenv()
from rag import ask_with_rag

def tts_pyttsx3(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS failed:", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["type","voice"], default="type")
    args = parser.parse_args()
    print("Voice RAG Agent started. Mode:", args.mode)
    while True:
        if args.mode == "type":
            q = input("You: ").strip()
            if q.lower() in ("quit","exit"):
                break
        else:
            # Voice mode not implemented in this minimal runner; use type mode or extend as needed.
            q = input("Type your question (voice mode not configured): ").strip()
        if not q:
            continue
        print("Thinking...")
        answer = ask_with_rag(q)
        print("Agent:", answer)
        try:
            tts_pyttsx3(answer)
        except Exception as e:
            print("TTS error:", e)

if __name__ == "__main__":
    main()
