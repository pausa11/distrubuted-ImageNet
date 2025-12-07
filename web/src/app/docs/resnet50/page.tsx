import Mermaid from "@/components/Mermaid";

export default function ResNet50Page() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">Arquitectura ResNet50</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    ResNet50 (Residual Network de 50 capas) es una de las arquitecturas de redes neuronales convolucionales (CNN) más influyentes en la visión por computadora.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">¿Por qué ImageNet1k?</h2>
                <p>
                    ImageNet1k es un subconjunto del dataset masivo ImageNet, que contiene aproximadamente 1.28 millones de imágenes de entrenamiento distribuidas en 1000 clases.
                    ResNet50 se ha convertido en el estándar de facto para benchmarks en este dataset debido a su equilibrio ideal entre precisión y costo computacional.
                    Es lo suficientemente compleja para aprender características ricas, pero lo suficientemente ligera para ser entrenada en tiempos razonables, lo que la hace perfecta para probar nuestra infraestructura distribuida.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">ResNet vs CNN Tradicionales</h2>
                <p>
                    En las redes convolucionales profundas tradicionales (como VGG), aumentar la profundidad no siempre mejora el rendimiento. De hecho, a menudo conduce al problema del <strong>gradiente desvanecido</strong>,
                    donde la señal de error se vuelve tan pequeña al propagarse hacia atrás que las primeras capas dejan de aprender.
                </p>
                <p>
                    ResNet resuelve esto introduciendo <strong>conexiones residuales</strong> (o "skip connections"). Estas conexiones permiten que la señal salte una o más capas,
                    facilitando el flujo del gradiente durante el entrenamiento. Matemáticamente, en lugar de aprender una función directa <code>H(x)</code>, las capas aprenden una función residual <code>F(x) = H(x) - x</code>,
                    lo que resulta ser más fácil de optimizar.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Visualización del Bloque Residual</h2>
                <p>
                    La unidad fundamental de ResNet es el bloque residual. A continuación se muestra cómo fluyen los datos a través de un bloque "bottleneck" típico de ResNet50:
                </p>

                <div className="my-8">
                    <Mermaid chart={`graph TD
    Input["Entrada x"] --> Conv1["Conv 1x1, BN, ReLU"]
    Conv1 --> Conv2["Conv 3x3, BN, ReLU"]
    Conv2 --> Conv3["Conv 1x1, BN"]
    
    Input -->|Skip Connection| AddNode((" + "))
    Conv3 --> AddNode
    
    AddNode --> ReLU["ReLU Final"]
    ReLU --> Output["Salida H(x)"]
    
    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#9f9,stroke:#333,stroke-width:2px
    style AddNode fill:#ff9,stroke:#333,stroke-width:2px`} />
                </div>
                <p className="text-sm text-gray-500 mt-2 text-center">
                    Figura 1: Estructura de un bloque residual "bottleneck". La conexión de salto permite que la información fluya sin obstáculos.
                </p>
            </div>
        </>
    );
}
