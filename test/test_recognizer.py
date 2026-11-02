
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition.recognizer import GestureRecognizer



def main():
    recognizer = GestureRecognizer()

    # Test input có đúng 63 features
    features = np.zeros(63, dtype=np.float32)

    result = recognizer.process(features)

    print("Recognizer test")
    print("--------------------")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
