# Algoritmo Gossip en Hivemind

## Descripción

Este proyecto implementa el **algoritmo Gossip** (o protocolo Gossip) para el entrenamiento distribuido de redes neuronales usando la biblioteca Hivemind. El algoritmo Gossip permite que múltiples peers (nodos) entrenen modelos de forma descentralizada sin necesidad de un coordinador central.

## ¿Qué es el Algoritmo Gossip?

El algoritmo Gossip es un protocolo de comunicación descentralizado inspirado en la forma en que se propagan los rumores en una red social. En el contexto del aprendizaje automático distribuido:

- **Comunicación peer-to-peer**: Cada nodo se comunica directamente con un subconjunto pequeño de otros nodos
- **Sin coordinador central**: No hay un servidor maestro que coordine el entrenamiento
- **Convergencia eventual**: A través de múltiples rondas de comunicación, todos los nodos convergen hacia el mismo modelo
- **Escalabilidad**: El patrón de comunicación permite agregar o eliminar nodos sin afectar el sistema completo
- **Tolerancia a fallos**: Si un nodo falla, los demás pueden continuar entrenando

## Implementación en Hivemind

Hivemind implementa el algoritmo Gossip a través de `DecentralizedAverager`, que es usado internamente por `hivemind.Optimizer`. Los componentes clave son:

### 1. DHT (Distributed Hash Table)
- Permite a los peers descubrirse mutuamente sin un coordinador central
- Mantiene un registro distribuido de peers activos

### 2. Local SGD con Gossip Averaging
- Cada peer entrena localmente usando SGD (Stochastic Gradient Descent)
- Periódicamente, los peers promedian sus parámetros con un grupo pequeño de otros peers
- Esto se logra con `use_local_updates=True`

### 3. Parámetros de Configuración del Gossip

Los scripts de entrenamiento exponen los siguientes parámetros configurables:

- `--gossip_group_size`: Tamaño del grupo de gossip (por defecto: 16)
  - Número objetivo de peers en cada ronda de averaging
  - Se recomienda usar potencias de 2 (2, 4, 8, 16, 32, etc.)
  - Grupos más grandes → mayor consenso pero más comunicación
  
- `--gossip_min_group_size`: Tamaño mínimo del grupo (por defecto: 2)
  - Número mínimo de peers necesarios para comenzar el averaging
  - Permite comenzar incluso con pocos peers disponibles
  
- `--gossip_alpha`: Tasa de aprendizaje para averaging (por defecto: 1.0)
  - Controla cuánto se mueven los parámetros hacia el promedio del grupo
  - 1.0 = promedio completo (parámetros = promedio del grupo)
  - 0.5 = promedio parcial (parámetros se mueven a medio camino hacia el promedio)
  - Valores menores pueden mejorar la estabilidad pero ralentizan la convergencia

## Uso

### Ejemplo Básico (CIFAR-10)

```bash
# Primer peer (bootstrap)
python src/trainGossipCifar.py \
  --host_maddr /ip4/0.0.0.0/tcp/4011 \
  --gossip_group_size 8 \
  --gossip_min_group_size 2 \
  --gossip_alpha 1.0

# Segundo peer (conecta al primero)
python src/trainGossipCifar.py \
  --initial_peer /ip4/127.0.0.1/tcp/4011/p2p/<PEER_ID> \
  --host_maddr /ip4/0.0.0.0/tcp/4012 \
  --gossip_group_size 8
```

### Ejemplo con ImageNet Tiny

```bash
# Primer peer
python src/trainGossipImagenet.py \
  --data_root ./data \
  --host_maddr /ip4/0.0.0.0/tcp/5001 \
  --gossip_group_size 16 \
  --gossip_min_group_size 2

# Segundo peer
python src/trainGossipImagenet.py \
  --data_root ./data \
  --initial_peer /ip4/127.0.0.1/tcp/5001/p2p/<PEER_ID> \
  --host_maddr /ip4/0.0.0.0/tcp/5002 \
  --gossip_group_size 16
```

## Ventajas del Algoritmo Gossip

1. **Escalabilidad**: Añadir más peers no sobrecarga la red de comunicación
2. **Tolerancia a fallos**: La falla de un peer no detiene el entrenamiento
3. **Descentralización**: No hay un único punto de falla
4. **Eficiencia**: Cada peer solo comunica con un subconjunto de otros peers
5. **Convergencia**: Todos los peers eventualmente convergen al mismo modelo

## Referencias

- [Hivemind Documentation](https://github.com/learning-at-home/hivemind)
- [Gossip-based Distributed Learning](https://arxiv.org/abs/1803.07068)
- [Decentralized Deep Learning with Gossip](https://arxiv.org/abs/1705.09056)
