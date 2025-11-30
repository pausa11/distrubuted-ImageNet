import Mermaid from "@/components/Mermaid";

export default function ArchitecturePage() {
    return (
        <>
            <h1 className="text-4xl font-bold text-gray-900 mb-6">Arquitectura del Sistema</h1>

            <div className="prose prose-blue max-w-none">
                <p className="lead text-xl text-gray-600">
                    El sistema está diseñado para ser totalmente descentralizado, tolerante a fallos y capaz de ejecutarse en hardware heterogéneo.
                </p>

                <h2 className="mt-8">Visión General de Alto Nivel</h2>
                <p>
                    La arquitectura consta de tres componentes principales:
                </p>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Nodos Trabajadores (Workers):</strong> Pares individuales que calculan gradientes y actualizan el modelo.</li>
                    <li><strong>DHT (Tabla Hash Distribuida):</strong> Una red peer-to-peer para encontrar otros pares y promediar gradientes.</li>
                    <li><strong>Almacenamiento (GCS):</strong> Almacenamiento centralizado para el dataset inmutable (ImageNet) y descubrimiento de pares (bootstrapping).</li>
                </ul>

                <h2 className="mt-8">Detalles de los Componentes</h2>

                <h3 className="text-xl font-semibold mt-6 mb-3">1. Nodos Trabajadores</h3>
                <p>
                    Cada trabajador ejecuta un bucle de entrenamiento PyTorch. El modelo (ResNet50) se replica en todos los trabajadores.
                    Los trabajadores calculan gradientes en su lote local de datos y luego colaboran con otros pares para promediar estos gradientes.
                </p>
                <p>
                    <strong>Tecnologías Clave:</strong> PyTorch, Hivemind, Torchvision.
                </p>

                <h3 className="text-xl font-semibold mt-6 mb-3">2. Optimización Descentralizada (Hivemind)</h3>
                <p>
                    En lugar de un servidor de parámetros central, utilizamos <code>hivemind.Optimizer</code>. Este optimizador envuelve al optimizador estándar de PyTorch
                    y maneja la comunicación. Encuentra pares óptimos en la DHT para promediar gradientes, asegurando que los parámetros del modelo
                    converjan a través de la red.
                </p>

                <h3 className="text-xl font-semibold mt-6 mb-3">3. Streaming de Datos (WebDataset)</h3>
                <p>
                    El dataset ImageNet es demasiado grande para descargarlo en cada trabajador. Utilizamos <strong>WebDataset</strong> para transmitir muestras de entrenamiento
                    directamente desde Google Cloud Storage (GCS). Los datos se fragmentan en miles de archivos <code>.tar</code>, permitiendo un acceso paralelo eficiente
                    sin el cuello de botella del sistema de archivos local.
                </p>

                <h2 className="mt-8">Diagrama de Arquitectura</h2>
                <div className="my-8">
                    <Mermaid chart={`graph TD
    subgraph Storage ["Google Cloud Storage"]
        DS["ImageNet Dataset (WebDataset)"]
        Boot["Peer Discovery Files"]
    end

    subgraph P2P_Network ["Hivemind DHT"]
        W1["Worker 1 (GPU)"]
        W2["Worker 2 (MPS)"]
        W3["Worker 3 (CPU)"]
    end

    DS -->|Stream Shards| W1
    DS -->|Stream Shards| W2
    DS -->|Stream Shards| W3

    W1 <-->|Gradient Averaging| W2
    W2 <-->|Gradient Averaging| W3
    W1 <-->|Gradient Averaging| W3

    W1 -.->|Announce/Fetch| Boot
    W2 -.->|Announce/Fetch| Boot
    W3 -.->|Announce/Fetch| Boot`} />
                </div>
                <p className="text-sm text-gray-500 mt-2 text-center">
                    Figura 1: Arquitectura de alto nivel mostrando el flujo de datos y la comunicación peer-to-peer.
                </p>
            </div>
        </>
    );
}
