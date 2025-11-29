import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <div className="text-center max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-5xl font-extrabold tracking-tight text-gray-900 sm:text-6xl mb-6">
          <span className="block">Project Documentation</span>
          <span className="block text-blue-600">Paper & Blog</span>
        </h1>
        <p className="mt-4 text-xl text-gray-500 max-w-2xl mx-auto">
          Welcome to the official documentation and blog for our project.
          Here you will find everything you need to know about our research and implementation.
        </p>
        <div className="mt-10 flex justify-center gap-4">
          <Link
            href="/docs"
            className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 md:py-4 md:text-lg md:px-10 transition-colors"
          >
            Get Started
          </Link>
          <Link
            href="/paper"
            className="px-8 py-3 border border-transparent text-base font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 md:py-4 md:text-lg md:px-10 transition-colors"
          >
            Read Paper
          </Link>
        </div>
      </div>

      <div className="mt-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 gap-8 md:grid-cols-3">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-medium text-gray-900">Documentation</h3>
          <p className="mt-2 text-gray-500">
            Comprehensive guides and API references to help you integrate and use our tools.
          </p>
          <Link href="/docs" className="mt-4 inline-block text-blue-600 hover:text-blue-500 font-medium">
            Learn more &rarr;
          </Link>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-medium text-gray-900">Research Paper</h3>
          <p className="mt-2 text-gray-500">
            Read our detailed research paper explaining the methodology and results.
          </p>
          <Link href="/paper" className="mt-4 inline-block text-blue-600 hover:text-blue-500 font-medium">
            Read now &rarr;
          </Link>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-medium text-gray-900">Blog</h3>
          <p className="mt-2 text-gray-500">
            Latest updates, tutorials, and insights from the team.
          </p>
          <Link href="/blog" className="mt-4 inline-block text-blue-600 hover:text-blue-500 font-medium">
            View posts &rarr;
          </Link>
        </div>
      </div>
    </div>
  );
}
