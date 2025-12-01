import Mermaid from "@/components/Mermaid";

export default function ImageNetPage() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">Dataset ImageNet1k</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    ImageNet es una base de datos de imágenes organizada según la jerarquía de WordNet. ImageNet1k es el subconjunto utilizado para el desafío ILSVRC 2012.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Estadísticas Clave</h2>
                <ul className="list-disc pl-6 space-y-2">
                    <li><strong>Clases:</strong> 1,000 categorías distintas.</li>
                    <li><strong>Imágenes de Entrenamiento:</strong> 1,281,167 imágenes.</li>
                    <li><strong>Imágenes de Validación:</strong> 50,000 imágenes (50 por clase).</li>
                    <li><strong>Resolución:</strong> Variable, pero típicamente redimensionada a 224x224 para ResNet.</li>
                </ul>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">El Desafío de la Diversidad</h2>
                <p>
                    Lo que hace difícil a ImageNet es la granularidad de sus clases. No solo debe distinguir entre "perro" y "gato", sino entre 120 razas diferentes de perros.
                    Esto obliga al modelo a aprender características extremadamente detalladas y específicas.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Jerarquía WordNet</h2>
                <p>
                    Las clases en ImageNet no son aleatorias; siguen una estructura semántica derivada de WordNet. Cada nodo en la jerarquía es un "synset" (conjunto de sinónimos).
                </p>

                <div className="my-8">
                    <Mermaid chart={`graph TD
    Root["Entidad"] --> Living["Ser Vivo"]
    Living --> Animal["Animal"]
    Animal --> Mammal["Mamífero"]
    Mammal --> Carnivore["Carnívoro"]
    Carnivore --> Canine["Canino"]
    Canine --> Dog["Perro"]
    Dog --> WorkingDog["Perro de Trabajo"]
    WorkingDog --> Husky["Husky Siberiano"]
    
    style Root fill:#e0e0e0,stroke:#333
    style Dog fill:#fff9c4,stroke:#fbc02d
    style Husky fill:#ffccbc,stroke:#d84315,stroke-width:2px`} />
                </div>
                <p className="text-sm text-gray-500 mt-2 text-center">
                    Figura 1: Ejemplo de la jerarquía de clases para "Husky Siberiano". El modelo debe aprender a clasificar la hoja específica del árbol.
                </p>
            </div>
        </>
    );
}
