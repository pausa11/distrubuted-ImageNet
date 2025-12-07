import Link from 'next/link';

export default function Navbar() {
    return (
        <nav className="bg-white/80 dark:bg-black/80 backdrop-blur-md border-b border-gray-100 dark:border-gray-800 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    <div className="flex-shrink-0 flex items-center">
                        <Link href="/" className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                            Async Distributed ImageNet1k
                        </Link>
                    </div>
                    <div className="hidden sm:flex sm:space-x-8">
                        <Link
                            href="/docs"
                            className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium"
                        >
                            Docs
                        </Link>
                        <Link
                            href="/statistics"
                            className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium"
                        >
                            Statistics
                        </Link>
                        <Link
                            href="/inference"
                            className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium"
                        >
                            Inference
                        </Link>
                    </div>
                </div>
            </div>
        </nav>
    );
}
