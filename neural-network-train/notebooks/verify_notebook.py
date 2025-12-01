
import sys
import os
import torch
import hivemind
import matplotlib.pyplot as plt
import numpy as np

# Agregar src al path
sys.path.append(os.path.abspath('../src'))

try:
    from dhtPeerImagenet import build_model
    from datasets import get_webdataset_loader
    print("✅ Importaciones exitosas")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

try:
    model = build_model(num_classes=1000)
    print("✅ Modelo creado exitosamente")
except Exception as e:
    print(f"❌ Error creando modelo: {e}")
    sys.exit(1)

print("Verificación básica completada.")
