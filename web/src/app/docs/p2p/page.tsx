import Mermaid from "@/components/Mermaid";

export default function P2PPage() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">Redes Peer-to-Peer (P2P)</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    Las redes Peer-to-Peer (P2P) representan un cambio de paradigma respecto a los modelos tradicionales centralizados, democratizando el acceso a recursos computacionales.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Concepto Fundamental</h2>
                <p>
                    En una red P2P, todos los participantes (nodos o "peers") son equipotentes. No existe una distinción jerárquica entre servidores y clientes.
                    Cada nodo actúa simultáneamente como cliente y servidor, consumiendo y proveyendo recursos (ancho de banda, almacenamiento, capacidad de cómputo) a la red.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Cliente-Servidor vs P2P</h2>
                <p>
                    El modelo tradicional <strong>Cliente-Servidor</strong> depende de una autoridad central que gestiona todos los recursos. Si bien es fácil de administrar,
                    presenta un punto único de fallo y cuellos de botella de escalabilidad: cuantos más clientes se unen, más carga soporta el servidor.
                </p>
                <p>
                    En contraste, las redes <strong>P2P</strong> son inherentemente escalables. Cuantos más nodos se unen, más recursos tiene la red disponible.
                    La carga se distribuye, eliminando cuellos de botella y aumentando la robustez ante fallos.
                </p>

                <div className="my-8">
                    <Mermaid chart={`graph TD
    subgraph CS ["Modelo Cliente-Servidor"]
        Server["Servidor Central"]
        C1["Cliente 1"]
        C2["Cliente 2"]
        C3["Cliente 3"]
        Server --> C1
        Server --> C2
        Server --> C3
    end

    subgraph P2P ["Modelo P2P (Malla)"]
        P1["Peer A"]
        P2["Peer B"]
        P3["Peer C"]
        P4["Peer D"]
        P1 <--> P2
        P2 <--> P3
        P3 <--> P4
        P4 <--> P1
        P1 <--> P3
    end
    
    style Server fill:#f9f,stroke:#333,stroke-width:2px
    style P1 fill:#9f9,stroke:#333,stroke-width:2px
    style P2 fill:#9f9,stroke:#333,stroke-width:2px
    style P3 fill:#9f9,stroke:#333,stroke-width:2px
    style P4 fill:#9f9,stroke:#333,stroke-width:2px`} />
                </div>
                <p className="text-sm text-gray-500 mt-2 text-center">
                    Figura 1: Comparación de topologías. El modelo P2P distribuye las conexiones, mientras que el Cliente-Servidor las centraliza.
                </p>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Relevancia en este Proyecto</h2>
                <p>
                    Para entrenar modelos masivos como ResNet50 en ImageNet, tradicionalmente se requeriría un clúster de supercomputación costoso.
                    Al adoptar una arquitectura P2P, podemos aprovechar el hardware ocioso de múltiples voluntarios (GPUs de consumo, laptops, etc.).
                    Hivemind actúa como el protocolo de coordinación que permite a estos nodos heterogéneos trabajar juntos como si fueran una sola supercomputadora virtual.
                </p>
            </div>
        </>
    );
}
