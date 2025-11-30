import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="text-center max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-5xl font-extrabold tracking-tight text-gray-900 sm:text-6xl mb-6">
          <span className="block">Entrenamiento Distribuido</span>
          <span className="block text-blue-600">Asíncrono de ImageNet1k</span>
        </h1>
        <p className="mt-4 text-xl text-gray-500 max-w-2xl mx-auto">
          Bienvenido a la documentación oficial y blog de nuestro proyecto.
          Aquí encontrarás todo lo que necesitas saber sobre nuestra investigación e implementación.
        </p>
        <div className="mt-10 flex justify-center gap-4">
          <Link
            href="/docs"
            className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 md:py-4 md:text-lg md:px-10 transition-colors"
          >
            Comenzar
          </Link>
        </div>
      </div>

      <div className="mt-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-center">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 max-w-md w-full">
          <h3 className="text-lg font-medium text-gray-900">Documentación</h3>
          <p className="mt-2 text-gray-500">
            Guías completas y referencias de API para ayudarte a integrar y usar nuestras herramientas.
          </p>
          <Link href="/docs" className="mt-4 inline-block text-blue-600 hover:text-blue-500 font-medium">
            Aprender más &rarr;
          </Link>
        </div>
      </div>
    </div>
  );
}
