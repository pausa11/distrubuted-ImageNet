# Entrenamiento Distribuido con Algoritmo Gossip

Este directorio contiene scripts para entrenar redes neuronales de forma distribuida usando el **algoritmo Gossip** implementado con [Hivemind](https://github.com/learning-at-home/hivemind).

## 📚 Contenido

- `src/trainGossipCifar.py` - Entrenamiento de CNN en CIFAR-10 con Gossip
- `src/trainGossipImagenet.py` - Entrenamiento de ResNet50 en ImageNet Tiny con Gossip
- `src/trainGossip.py` - Script base para Tiny ImageNet
- `GOSSIP_ALGORITHM.md` - Documentación detallada del algoritmo Gossip
- `test_gossip_args.py` - Tests unitarios para argumentos
- `test_gossip_config.py` - Test de integración

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar con CIFAR-10 (Ejemplo Simple)

**Terminal 1 - Primer Peer (Bootstrap):**
```bash
python src/trainGossipCifar.py \
  --host_maddr /ip4/0.0.0.0/tcp/4011 \
  --batch 64 \
  --epochs 10 \
  --gossip_group_size 8
```

El script imprimirá algo como:
```
VISIBLE_MADDRS: ['/ip4/192.168.1.100/tcp/4011/p2p/QmXXXXXX...']
```

**Terminal 2 - Segundo Peer:**
```bash
python src/trainGossipCifar.py \
  --initial_peer /ip4/192.168.1.100/tcp/4011/p2p/QmXXXXXX... \
  --host_maddr /ip4/0.0.0.0/tcp/4012 \
  --batch 64 \
  --epochs 10 \
  --gossip_group_size 8
```

**Terminal 3 - Tercer Peer:**
```bash
python src/trainGossipCifar.py \
  --initial_peer /ip4/192.168.1.100/tcp/4011/p2p/QmXXXXXX... \
  --host_maddr /ip4/0.0.0.0/tcp/4013 \
  --batch 64 \
  --epochs 10 \
  --gossip_group_size 8
```

## ⚙️ Parámetros del Algoritmo Gossip

### `--gossip_group_size` (default: 16)
Tamaño objetivo del grupo para averaging. Determina cuántos peers intentarán promediar sus parámetros juntos en cada ronda.

- **Valores recomendados**: Potencias de 2 (4, 8, 16, 32)
- **Grupos pequeños (4-8)**: Menos comunicación, convergencia más lenta
- **Grupos grandes (16-32)**: Más comunicación, convergencia más rápida

**Ejemplo:**
```bash
--gossip_group_size 8  # Grupos de hasta 8 peers
```

### `--gossip_min_group_size` (default: 2)
Número mínimo de peers necesarios para comenzar el averaging.

- **Valor bajo (2)**: Permite entrenar con pocos peers
- **Valor alto (4-8)**: Requiere más peers, pero averaging más estable

**Ejemplo:**
```bash
--gossip_min_group_size 2  # Comenzar con al menos 2 peers
```

### `--gossip_alpha` (default: 1.0)
Factor de averaging que controla cuánto se mueven los parámetros locales hacia el promedio del grupo.

- **alpha = 1.0**: Promedio completo (parámetros locales = promedio del grupo)
- **alpha = 0.5**: Promedio parcial (parámetros se mueven a medio camino)
- **alpha < 1.0**: Mayor estabilidad, convergencia más lenta

**Ejemplo:**
```bash
--gossip_alpha 0.9  # Averaging suave
```

## 📊 Ejemplos de Uso

### CIFAR-10 con 3 Peers

```bash
# Peer 1
python src/trainGossipCifar.py \
  --host_maddr /ip4/0.0.0.0/tcp/4011 \
  --gossip_group_size 4 \
  --gossip_min_group_size 2 \
  --gossip_alpha 1.0 \
  --batch 64 \
  --lr 0.01 \
  --epochs 20

# Peer 2 (usar la VISIBLE_MADDR del Peer 1)
python src/trainGossipCifar.py \
  --initial_peer /ip4/127.0.0.1/tcp/4011/p2p/<PEER_ID> \
  --host_maddr /ip4/0.0.0.0/tcp/4012 \
  --gossip_group_size 4 \
  --batch 64 \
  --lr 0.01 \
  --epochs 20

# Peer 3
python src/trainGossipCifar.py \
  --initial_peer /ip4/127.0.0.1/tcp/4011/p2p/<PEER_ID> \
  --host_maddr /ip4/0.0.0.0/tcp/4013 \
  --gossip_group_size 4 \
  --batch 64 \
  --lr 0.01 \
  --epochs 20
```

### ImageNet Tiny con Averaging Agresivo

```bash
# Configuración para convergencia rápida con muchos peers
python src/trainGossipImagenet.py \
  --data_root ./data \
  --host_maddr /ip4/0.0.0.0/tcp/5001 \
  --gossip_group_size 32 \
  --gossip_min_group_size 8 \
  --gossip_alpha 1.0 \
  --batch 32 \
  --lr 0.1 \
  --epochs 20
```

### Averaging Conservador (para experimentos)

```bash
# Configuración con averaging más suave
python src/trainGossipCifar.py \
  --host_maddr /ip4/0.0.0.0/tcp/4011 \
  --gossip_group_size 8 \
  --gossip_min_group_size 2 \
  --gossip_alpha 0.7 \
  --batch 64 \
  --epochs 30
```

## 🔬 Tests

### Ejecutar Tests de Argumentos
```bash
python test_gossip_args.py
```

Este test verifica que:
- Los argumentos de gossip se parsean correctamente
- Los valores por defecto son correctos
- La configuración se pasa a `gossip_averager_opts`

## 📖 Más Información

Para una explicación detallada del algoritmo Gossip y cómo funciona en Hivemind, consulta:

- **[GOSSIP_ALGORITHM.md](GOSSIP_ALGORITHM.md)** - Documentación completa del algoritmo
- **[Hivemind Documentation](https://github.com/learning-at-home/hivemind)** - Documentación oficial de Hivemind

## 🎯 Ventajas del Algoritmo Gossip

1. **✅ Escalabilidad**: Agregar más peers no sobrecarga la red
2. **✅ Tolerancia a fallos**: Un peer que falla no detiene el entrenamiento
3. **✅ Descentralización**: No hay un único punto de falla
4. **✅ Eficiencia**: Comunicación peer-to-peer directa
5. **✅ Convergencia**: Todos los peers convergen al mismo modelo

## 🐛 Troubleshooting

### Error: "Daemon failed to start"
- Asegúrate de tener instalados todos los componentes de hivemind
- Verifica que los puertos no estén en uso
- Intenta usar puertos diferentes con `--host_maddr`

### Los peers no se encuentran
- Verifica que la dirección `--initial_peer` sea correcta (debe incluir `/p2p/<PEER_ID>`)
- Asegúrate de que no haya firewalls bloqueando la comunicación
- Verifica que estés usando la IP correcta (usa la IP local si estás en la misma máquina)

### Averaging muy lento
- Aumenta `--gossip_group_size` para averaging más frecuente
- Reduce `--gossip_alpha` para hacer el averaging más gradual
- Asegúrate de tener suficientes peers activos

## 📝 Notas

- Los checkpoints se guardan automáticamente en `checkpoints/`
- El mejor modelo se guarda con sufijo `_best.pt`
- El modelo final se guarda con sufijo `_final.pt`
- Usa `--resume_from` para continuar desde un checkpoint

## 🤝 Contribuciones

Para contribuir o reportar issues, visita el repositorio principal.
