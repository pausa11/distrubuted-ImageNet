export default function QuickStartPage() {
    return (
        <>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white mb-8 tracking-tight">Inicio Rápido</h1>

            <div className="prose prose-blue dark:prose-invert max-w-none prose-headings:font-bold prose-a:text-blue-600 dark:prose-a:text-blue-400">
                <p className="lead text-xl text-gray-600">
                    Pon en marcha tu nodo de entrenamiento distribuido en minutos.
                </p>

                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 my-6">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <p className="text-sm text-yellow-700">
                                Asegúrate de haber completado los pasos de <a href="/docs/installation" className="font-medium underline hover:text-yellow-600">Instalación</a> primero.
                            </p>
                        </div>
                    </div>
                </div>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Opción 1: Iniciar como Par Inicial (Bootstrap)</h2>
                <p>
                    Si eres el primer nodo en la red, necesitas iniciar como el par inicial.
                    Opcionalmente puedes anunciar tu dirección a GCS para que otros puedan encontrarte.
                </p>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`python src/dhtPeerImagenet.py \\
    --announce_gcs_path gs://your-bucket/initial_peer.txt \\
    --batch_size 32`}
                    </pre>
                </div>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Opción 2: Iniciar como Par Trabajador (Worker)</h2>
                <p>
                    La mayoría de los nodos se unirán como trabajadores. Necesitas saber la dirección de un par existente,
                    o obtenerla de GCS si fue anunciada.
                </p>

                <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mt-8 mb-3">Método A: Auto-descubrimiento vía GCS</h3>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`python src/dhtPeerImagenet.py \\
    --fetch_gcs_path gs://your-bucket/initial_peer.txt \\
    --batch_size 32`}
                    </pre>
                </div>

                <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mt-8 mb-3">Método B: Dirección de Par Manual</h3>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`python src/dhtPeerImagenet.py \\
    --initial_peer /ip4/1.2.3.4/tcp/31337/p2p/QmHash... \\
    --batch_size 32`}
                    </pre>
                </div>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Argumentos Comunes</h2>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Argumento</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Descripción</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Por Defecto</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200 text-sm">
                            <tr>
                                <td className="px-6 py-4 font-mono">--batch_size</td>
                                <td className="px-6 py-4">Tamaño de lote local por paso.</td>
                                <td className="px-6 py-4">64</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 font-mono">--target_batch_size</td>
                                <td className="px-6 py-4">Tamaño de lote objetivo global para promediar.</td>
                                <td className="px-6 py-4">50000</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 font-mono">--device</td>
                                <td className="px-6 py-4">Forzar dispositivo (cpu, cuda, mps).</td>
                                <td className="px-6 py-4">Auto</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 font-mono">--num_workers</td>
                                <td className="px-6 py-4">Trabajadores de carga de datos. Usar 0 para MPS.</td>
                                <td className="px-6 py-4">0</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <h2 className="text-3xl font-bold text-gray-800 dark:text-gray-100 mt-10 mb-4">Monitoreo</h2>
                <p>
                    El script de entrenamiento registra métricas en el directorio <code>stats/</code>.
                    Puedes monitorear el progreso revisando los archivos CSV generados o la salida estándar.
                </p>
            </div>
        </>
    );
}
