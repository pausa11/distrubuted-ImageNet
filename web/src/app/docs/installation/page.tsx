export default function InstallationPage() {
    return (
        <>
            <h1 className="text-4xl font-bold text-gray-900 mb-6">Instalación</h1>

            <div className="prose prose-blue max-w-none">
                <p className="lead text-xl text-gray-600">
                    Sigue estos pasos para configurar el entorno y comenzar a contribuir al entrenamiento distribuido.
                </p>

                <h2 className="mt-8">Prerrequisitos</h2>
                <ul className="list-disc pl-6 space-y-2">
                    <li>Python 3.8 o superior</li>
                    <li>pip (instalador de paquetes de Python)</li>
                    <li>Git</li>
                    <li>Acceso a Google Cloud Storage (opcional, para streaming de datos)</li>
                </ul>

                <h2 className="mt-8">Paso 1: Clonar el Repositorio</h2>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`git clone https://github.com/your-username/distributed-imagenet.git
cd distributed-imagenet`}
                    </pre>
                </div>

                <h2 className="mt-8">Paso 2: Crear un Entorno Virtual</h2>
                <p>Se recomienda usar un entorno virtual para gestionar las dependencias.</p>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`# Create virtual environment
python -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Activate it (Windows)
.venv\\Scripts\\activate`}
                    </pre>
                </div>

                <h2 className="mt-8">Paso 3: Instalar Dependencias</h2>
                <p>Instala los paquetes requeridos usando pip:</p>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`pip install -r requirements.txt`}
                    </pre>
                </div>

                <h3 className="text-xl font-semibold mt-6 mb-3">Dependencias Clave</h3>
                <ul className="list-disc pl-6 space-y-2">
                    <li><code>torch &gt;= 2.2</code>: Framework de deep learning PyTorch.</li>
                    <li><code>torchvision &gt;= 0.17</code>: Datasets y modelos de imagen.</li>
                    <li><code>hivemind == 1.1.11</code>: Librería de deep learning descentralizado.</li>
                    <li><code>google-cloud-storage</code>: Para acceder al dataset ImageNet.</li>
                </ul>

                <h2 className="mt-8">Paso 4: Verificar Instalación</h2>
                <p>Ejecuta el siguiente comando para verificar si todo se instaló correctamente:</p>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                    <pre className="text-sm font-mono">
                        {`python -c "import torch; import hivemind; print('Success!')"`}
                    </pre>
                </div>
            </div>
        </>
    );
}
