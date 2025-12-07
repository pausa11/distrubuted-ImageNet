import Mermaid from "@/components/Mermaid";

export default function WebDatasetPage() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">WebDataset y Streaming</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    WebDataset es una biblioteca diseñada para el entrenamiento de alto rendimiento con datasets masivos almacenados en la nube.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">El Problema de I/O</h2>
                <p>
                    ImageNet contiene más de 1.2 millones de imágenes. Si almacenamos cada imagen como un archivo individual, el sistema de archivos sufre enormemente.
                    Durante el entrenamiento, leer millones de archivos pequeños genera una latencia de acceso aleatorio (seek time) que se convierte en el cuello de botella, dejando a la GPU esperando por datos.
                    Además, descargar todo el dataset (150GB+) en cada nodo de un voluntario es impráctico.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">La Solución: Sharding y Streaming</h2>
                <p>
                    WebDataset resuelve esto empaquetando miles de imágenes y sus etiquetas en archivos <code>.tar</code> grandes, llamados "shards".
                    Esto permite lecturas secuenciales de alto ancho de banda, lo cual es ideal para discos mecánicos y conexiones de red.
                </p>
                <p>
                    En nuestro sistema, los shards se almacenan en Google Cloud Storage (GCS). Los trabajadores no descargan el dataset completo; en su lugar,
                    hacen <strong>streaming</strong> de shards aleatorios bajo demanda. Esto permite comenzar a entrenar en segundos, sin importar el tamaño total del dataset.
                </p>

                <div className="my-8">
                    <Mermaid chart={`graph LR
    subgraph Cloud ["Google Cloud Storage"]
        Shard1["Shard 001.tar"]
        Shard2["Shard 002.tar"]
        ShardN["... Shard N.tar"]
    end

    subgraph Worker ["Nodo Trabajador"]
        Stream["Streamer"]
        Buffer["Shuffle Buffer"]
        Batch["Batch Assembler"]
        GPU["GPU Training"]
    end

    Shard1 -.->|HTTP Stream| Stream
    Shard2 -.->|HTTP Stream| Stream
    
    Stream -->|Extract Samples| Buffer
    Buffer -->|Randomize| Batch
    Batch -->|Tensor Batch| GPU
    
    style Cloud fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Worker fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style GPU fill:#4caf50,stroke:#1b5e20,stroke-width:2px,color:white`} />
                </div>
                <p className="text-sm text-gray-500 mt-2 text-center">
                    Figura 1: Pipeline de streaming de datos. Los archivos tar se transmiten, descomprimen y mezclan en memoria (shuffle buffer) antes de llegar a la GPU.
                </p>
            </div>
        </>
    );
}
