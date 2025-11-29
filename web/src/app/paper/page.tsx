export default function PaperPage() {
    return (
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="text-center mb-12">
                <h1 className="text-4xl font-bold text-gray-900 mb-4">Research Paper</h1>
                <p className="text-xl text-gray-500">
                    Distributed ImageNet Training: A New Approach
                </p>
            </div>

            <div className="bg-white shadow-sm border border-gray-200 rounded-xl p-8 md:p-12">
                <div className="prose prose-lg max-w-none">
                    <h2>Abstract</h2>
                    <p>
                        We present a novel approach to distributed training of neural networks on the ImageNet dataset.
                        Our method leverages...
                    </p>

                    <div className="my-8 p-4 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-800 text-sm">
                        <strong>Note:</strong> This is a placeholder for the actual paper content.
                        You can embed a PDF here or render the LaTeX content.
                    </div>

                    <h2>1. Introduction</h2>
                    <p>
                        The scale of modern datasets requires distributed training...
                    </p>

                    <div className="mt-12 flex justify-center">
                        <button className="px-6 py-3 bg-gray-900 text-white rounded-md hover:bg-gray-800 transition-colors flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                            Download PDF
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
