import sys
import os

try:
    import surprise
    import pandas as pd
    print("✅ Bravo ! Surprise et Pandas sont bien détectés.")
    print(f"📍 Python utilisé : {sys.executable}")
except ImportError as e:
    print(f"❌ Erreur : {e}")
    print("💡 L'environnement sélectionné n'est probablement pas le bon.")