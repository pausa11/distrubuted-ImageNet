export default function DocsPage() {
    return (
        <div className="flex min-h-[calc(100vh-4rem)]">
            {/* Sidebar */}
            <aside className="w-64 bg-white border-r border-gray-200 hidden md:block">
                <div className="p-6">
                    <h3 className="font-semibold text-gray-900 uppercase tracking-wider text-sm mb-4">
                        Getting Started
                    </h3>
                    <ul className="space-y-3">
                        <li>
                            <a href="#" className="text-blue-600 font-medium block">Introduction</a>
                        </li>
                        <li>
                            <a href="#" className="text-gray-600 hover:text-gray-900 block">Installation</a>
                        </li>
                        <li>
                            <a href="#" className="text-gray-600 hover:text-gray-900 block">Quick Start</a>
                        </li>
                    </ul>

                    <h3 className="font-semibold text-gray-900 uppercase tracking-wider text-sm mt-8 mb-4">
                        Core Concepts
                    </h3>
                    <ul className="space-y-3">
                        <li>
                            <a href="#" className="text-gray-600 hover:text-gray-900 block">Architecture</a>
                        </li>
                        <li>
                            <a href="#" className="text-gray-600 hover:text-gray-900 block">Data Flow</a>
                        </li>
                    </ul>
                </div>
            </aside>

            {/* Content */}
            <div className="flex-1 p-8 md:p-12">
                <div className="max-w-3xl">
                    <h1 className="text-4xl font-bold text-gray-900 mb-6">Introduction</h1>
                    <p className="text-lg text-gray-600 mb-8">
                        Welcome to the documentation. This project aims to solve X by doing Y.
                        Here you will find detailed instructions on how to set up, configure, and use the system.
                    </p>

                    <div className="prose prose-blue max-w-none">
                        <h2>Overview</h2>
                        <p>
                            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
                        </p>

                        <h2>Key Features</h2>
                        <ul>
                            <li>Feature 1: Description of feature 1.</li>
                            <li>Feature 2: Description of feature 2.</li>
                            <li>Feature 3: Description of feature 3.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
}
