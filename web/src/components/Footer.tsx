export default function Footer() {
    return (
        <footer className="bg-white border-t border-gray-100 mt-auto">
            <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
                <p className="text-center text-gray-500 text-sm">
                    &copy; {new Date().getFullYear()} ProjectName. All rights reserved.
                </p>
            </div>
        </footer>
    );
}
