#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv()

def get_threshold():
    return float(os.getenv("SIMILARITY_THRESHOLD", 0.65))
