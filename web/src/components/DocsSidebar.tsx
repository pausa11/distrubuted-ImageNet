import Link from 'next/link';

export default function DocsSidebar() {
    return (
        <aside className="w-64 bg-white border-r border-gray-200 hidden md:block h-[calc(100vh-4rem)] sticky top-16 overflow-y-auto">
            <div className="p-6">
                <h3 className="font-semibold text-gray-900 uppercase tracking-wider text-sm mb-4">
                    Comenzando
                </h3>
                <ul className="space-y-3">
                    <li>
                        <Link href="/docs" className="text-gray-600 hover:text-blue-600 block transition-colors">
                            Introducción
                        </Link>
                    </li>
                    <li>
                        <Link href="/docs/installation" className="text-gray-600 hover:text-blue-600 block transition-colors">
                            Instalación
                        </Link>
                    </li>
                    <li>
                        <Link href="/docs/quickstart" className="text-gray-600 hover:text-blue-600 block transition-colors">
                            Inicio Rápido
                        </Link>
                    </li>
                </ul>

                <h3 className="font-semibold text-gray-900 uppercase tracking-wider text-sm mt-8 mb-4">
                    Conceptos Principales
                </h3>
                <ul className="space-y-3">
                    <li>
                        <Link href="/docs/architecture" className="text-gray-600 hover:text-blue-600 block transition-colors">
                            Arquitectura
                        </Link>
                    </li>
                    <li>
                        <Link href="/docs/data-flow" className="text-gray-600 hover:text-blue-600 block transition-colors">
                            Flujo de Datos
                        </Link>
                    </li>
                </ul>
            </div>
        </aside>
    );
}
