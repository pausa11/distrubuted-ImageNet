'use client';

import { useChat } from '@ai-sdk/react';
import { UIMessage } from 'ai';
import { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Minimize2, Maximize2 } from 'lucide-react';

export default function Chatbot() {
    const { messages, sendMessage, status, error } = useChat({
        onError: (err) => {
            console.error("Chat error:", err);
        },
        onFinish: (message) => {
            console.log("Chat finished:", message);
        }
    });

    console.log("Chat status:", status);
    console.log("Chat error object:", error);

    const [input, setInput] = useState('');
    const [isOpen, setIsOpen] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const isLoading = status === 'submitted' || status === 'streaming';

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInput(e.target.value);
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        sendMessage({ text: input });
        setInput('');
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const getMessageText = (m: any) => {
        if (typeof m.content === 'string') return m.content;
        if (Array.isArray(m.parts)) {
            return m.parts
                .filter((part: any) => part.type === 'text')
                .map((part: any) => part.text)
                .join('');
        }
        return '';
    };

    return (
        <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end">
            {/* Chat Window */}
            {isOpen && (
                <div
                    className={`bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl overflow-hidden flex flex-col mb-4 transition-all duration-300 ease-in-out ${isExpanded ? 'w-[80vw] h-[80vh]' : 'w-80 h-96'
                        }`}
                >
                    {/* Header */}
                    <div className="bg-blue-600 p-3 flex justify-between items-center text-white">
                        <h3 className="font-semibold">Neural Network Assistant</h3>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setIsExpanded(!isExpanded)}
                                className="hover:bg-blue-700 p-1 rounded"
                            >
                                {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                            </button>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="hover:bg-blue-700 p-1 rounded"
                            >
                                <X size={16} />
                            </button>
                        </div>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {messages.length === 0 && (
                            <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
                                <p>Hello! I have context of the neural-network-train project.</p>
                                <p className="text-sm mt-2">Ask me anything about the codebase!</p>
                            </div>
                        )}
                        {messages.map((m: any) => (
                            <div
                                key={m.id}
                                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'
                                    }`}
                            >
                                <div
                                    className={`max-w-[80%] p-3 rounded-lg text-sm ${m.role === 'user'
                                        ? 'bg-blue-600 text-white rounded-br-none'
                                        : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded-bl-none'
                                        }`}
                                >
                                    <p className="whitespace-pre-wrap">{getMessageText(m)}</p>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-lg rounded-bl-none">
                                    <div className="flex space-x-2">
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSubmit} className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                        <div className="flex gap-2">
                            <input
                                className="flex-1 p-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm"
                                value={input}
                                onChange={handleInputChange}
                                placeholder="Ask about the project..."
                            />
                            <button
                                type="submit"
                                disabled={isLoading || !input.trim()}
                                className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                <Send size={18} />
                            </button>
                        </div>
                    </form>
                </div>
            )}

            {/* Toggle Button */}
            {!isOpen && (
                <button
                    onClick={() => setIsOpen(true)}
                    className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg transition-transform hover:scale-110 flex items-center justify-center"
                >
                    <MessageCircle size={24} />
                </button>
            )}
        </div>
    );
}
