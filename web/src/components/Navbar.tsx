import Link from 'next/link';

export default function Navbar() {
    return (
        <nav className="bg-white/80 backdrop-blur-md border-b border-gray-100 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    <div className="flex-shrink-0 flex items-center">
                        <Link href="/" className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                            ProjectName
                        </Link>
                    </div>
                    <div className="hidden sm:flex sm:space-x-8">
                        <Link href="/docs" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                            Documentation
                        </Link>
                        <Link href="/paper" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                            Paper
                        </Link>
                        <Link href="/blog" className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors">
                            Blog
                        </Link>
                    </div>
                </div>
            </div>
        </nav>
    );
}
