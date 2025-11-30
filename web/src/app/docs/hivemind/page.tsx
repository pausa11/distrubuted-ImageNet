export default function HivemindPage() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">Profundizando en Hivemind</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    Hivemind es la columna vertebral de nuestra arquitectura descentralizada. Permite entrenar modelos grandes a través de internet sin requerir una conexión de red de baja latencia.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Tabla Hash Distribuida (DHT)</h2>
                <p>
                    Utilizamos una implementación de Kademlia para formar una red superpuesta. Cada nodo en la red tiene un ID único y es responsable de una parte del espacio de claves.
                    Esto permite a los nuevos pares descubrirse entre sí y encontrar vecinos para colaborar en el entrenamiento sin necesidad de un servidor central de registro.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Promediado Descentralizado</h2>
                <p>
                    Para sincronizar los pesos del modelo, los trabajadores realizan un proceso de "All-Reduce" descentralizado. En lugar de enviar todos los gradientes a un servidor central,
                    los pares intercambian y promedian gradientes con sus vecinos en la DHT. Este proceso de chismes (gossiping) asegura que la información se propague exponencialmente rápido a través de la red.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Tolerancia a Fallos</h2>
                <p>
                    El sistema está diseñado para soportar una alta tasa de rotación (churn). Si un nodo se desconecta en medio de un cálculo, la DHT se reconfigura automáticamente
                    y el entrenamiento continúa con los nodos restantes. No hay un punto único de fallo.
                </p>
            </div>
        </>
    );
}
