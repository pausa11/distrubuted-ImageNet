'use client';

import { useState, useRef } from 'react';
import { predictImage, PredictionResult } from '../actions/predict';
import Image from 'next/image';

export default function InferencePage() {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [file, setFile] = useState<File | null>(null);
    const [prediction, setPrediction] = useState<PredictionResult | null>(null);
    const [loading, setLoading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const f = e.target.files[0];
            setFile(f);
            setSelectedImage(URL.createObjectURL(f));
            setPrediction(null);
        }
    };

    const handlePredict = async () => {
        if (!file) return;

        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('image', file);
            const result = await predictImage(formData);
            setPrediction(result);
        } catch (error) {
            console.error(error);
            setPrediction({ class_index: -1, synset: '', confidence: 0, error: 'Failed to predict' });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 p-8">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-3xl font-bold mb-8 text-blue-400">Model Inference</h1>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Upload Section */}
                    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
                        <h2 className="text-xl font-semibold mb-4">Upload Image</h2>

                        <div
                            className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                type="file"
                                ref={fileInputRef}
                                className="hidden"
                                accept="image/*"
                                onChange={handleFileSelect}
                            />
                            {selectedImage ? (
                                <div className="relative aspect-square w-full max-w-xs mx-auto overflow-hidden rounded-lg">
                                    <Image
                                        src={selectedImage}
                                        alt="Selected"
                                        fill
                                        className="object-cover"
                                    />
                                </div>
                            ) : (
                                <div className="text-gray-400">
                                    <p className="mb-2">Click to select an image</p>
                                    <p className="text-sm">JPG, PNG supported</p>
                                </div>
                            )}
                        </div>

                        <button
                            onClick={handlePredict}
                            disabled={!file || loading}
                            className={`mt-6 w-full py-3 px-4 rounded-lg font-medium transition-colors ${!file || loading
                                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                                }`}
                        >
                            {loading ? 'Processing...' : 'Run Prediction'}
                        </button>
                    </div>

                    {/* Results Section */}
                    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-lg">
                        <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>

                        {prediction ? (
                            <div className="space-y-6">
                                {prediction.error ? (
                                    <div className="p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
                                        {prediction.error}
                                    </div>
                                ) : (
                                    <>
                                        <div className="text-center p-6 bg-gray-900/50 rounded-lg border border-gray-700">
                                            <p className="text-sm text-gray-400 mb-1">Predicted Class</p>
                                            <p className="text-3xl font-bold text-white mb-2">{prediction.synset}</p>
                                            <div className="inline-block px-3 py-1 bg-blue-900/50 text-blue-300 rounded-full text-sm border border-blue-800">
                                                Index: {prediction.class_index}
                                            </div>
                                        </div>

                                        <div>
                                            <div className="flex justify-between text-sm mb-2">
                                                <span className="text-gray-400">Confidence</span>
                                                <span className="text-blue-400 font-mono">{(prediction.confidence * 100).toFixed(2)}%</span>
                                            </div>
                                            <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-blue-600 to-purple-600 transition-all duration-1000 ease-out"
                                                    style={{ width: `${prediction.confidence * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        ) : (
                            <div className="h-full flex items-center justify-center text-gray-500">
                                <p>Run inference to see results</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
