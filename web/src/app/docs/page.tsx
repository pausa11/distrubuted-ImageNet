export default function DocsPage() {
    return (
        <>
            <h1 className="text-4xl font-bold text-gray-900 mb-6">Introducción</h1>
            <p className="text-lg text-gray-600 mb-8">
                Bienvenido a la documentación del proyecto <strong>Entrenamiento Distribuido Asíncrono de ImageNet1k</strong>.
                Este sistema permite el entrenamiento descentralizado de modelos ResNet50 en el dataset ImageNet utilizando hardware heterogéneo.
            </p>

            <div className="prose prose-blue max-w-none">
                <h2>Visión General</h2>
                <p>
                    Nuestro proyecto aprovecha la librería <code>hivemind</code> para distribuir el proceso de entrenamiento entre múltiples pares.
                    A diferencia del entrenamiento distribuido tradicional que requiere un clúster de alto ancho de banda, nuestro enfoque funciona a través de internet
                    con hardware de consumo.
                </p>

                <h2>Características Principales</h2>
                <ul>
                    <li><strong>Entrenamiento Descentralizado:</strong> Sin servidor de parámetros central; los pares se comunican vía DHT.</li>
                    <li><strong>Tolerancia a Fallos:</strong> Los pares pueden unirse y salir en cualquier momento sin detener el entrenamiento.</li>
                    <li><strong>Streaming de Datos Eficiente:</strong> Utiliza WebDataset para transmitir datos de ImageNet desde Google Cloud Storage.</li>
                    <li><strong>Hardware Heterogéneo:</strong> Soporta CUDA, MPS (Apple Silicon) y CPU.</li>
                </ul>
            </div>
        </>
    );
}
