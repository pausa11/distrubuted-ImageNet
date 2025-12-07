import Mermaid from "@/components/Mermaid";

export default function DataFlowPage() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">Flujo de Datos</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    Entendiendo cómo 1.2 millones de imágenes son transmitidas, procesadas y entrenadas sin un dataset local.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">El Pipeline de Streaming</h2>
                <p>
                    El deep learning tradicional requiere descargar todo el dataset a un disco local.
                    Para ImageNet (150GB+), esto es impráctico para nodos transitorios en la nube o dispositivos personales.
                    Nuestro sistema utiliza <strong>WebDataset</strong> para transmitir datos directamente desde Google Cloud Storage (GCS) a la memoria de la GPU.
                </p>

                <div className="my-8">
                    <Mermaid chart={`graph LR
    GCS["Google Cloud Storage"] -->|Stream HTTP| Shards["Shards Tar"]
    Shards -->|Decodificar| Images["Imágenes PIL"]
    Images -->|Transformar| Tensors["Tensores PyTorch"]
    Tensors -->|Batch| Batch["Mini-Batch"]
    Batch -->|Prefetch| GPU["Memoria GPU/MPS"]`} />
                </div>
                <p className="text-sm text-gray-500 mt-2 text-center">
                    Figura 1: El pipeline de streaming de datos desde la Nube a la GPU.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Componentes Clave</h2>

                <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mt-8 mb-3">1. Shards de WebDataset</h3>
                <p>
                    El dataset ImageNet es pre-procesado en cientos de archivos <code>.tar</code> (shards).
                    Cada shard contiene ~1000 imágenes y sus etiquetas de clase. Este formato es óptimo para lectura secuencial
                    y evita la sobrecarga de millones de pequeñas solicitudes de archivos.
                </p>

                <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mt-8 mb-3">2. Procesamiento al Vuelo</h3>
                <p>
                    A medida que los datos se transmiten:
                </p>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Decodificación:</strong> Los bytes crudos se decodifican en imágenes.</li>
                    <li><strong>Aumentación:</strong> Recortes aleatorios, volteos y normalización se aplican en tiempo real.</li>
                    <li><strong>Mezcla (Shuffle):</strong> Un buffer local mezcla las muestras para asegurar aleatoriedad en el entrenamiento.</li>
                </ul>

                <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mt-8 mb-3">3. Optimizaciones para Hardware Heterogéneo</h3>
                <p>
                    Diferente hardware requiere diferentes estrategias de E/S:
                </p>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>CUDA (Linux):</strong> Utiliza <code>DataLoader</code> estándar de PyTorch con múltiples trabajadores para procesamiento paralelo.</li>
                    <li><strong>MPS (macOS):</strong> Utiliza un <code>ThreadedWebDataset</code> personalizado para evitar limitaciones de multiprocesamiento en Apple Silicon, usando hilos para pre-cargar datos sin bloquear el bucle de entrenamiento principal.</li>
                </ul>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Eficiencia de Red</h2>
                <p>
                    Para manejar la inestabilidad de la red:
                </p>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Reintentos:</strong> El cliente de streaming reintenta automáticamente las conexiones fallidas.</li>
                    <li><strong>Caché:</strong> (Opcional) Los shards pueden ser cacheados localmente para reducir el uso de ancho de banda en épocas subsiguientes.</li>
                    <li><strong>Robustez:</strong> Si un shard falla completamente, el sistema registra una advertencia y pasa al siguiente, asegurando que el entrenamiento no se detenga.</li>
                </ul>
            </div>
        </>
    );
}
