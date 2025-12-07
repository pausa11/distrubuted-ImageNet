'use client';

import { useState, useEffect } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';
import { getStatsRuns, getRunData, RunData } from '@/app/actions/stats';

interface StatsViewerProps {
    folders: string[];
}

export default function StatsViewer({ folders }: StatsViewerProps) {
    const [selectedFolder, setSelectedFolder] = useState<string>(folders[0] || '');
    const [runs, setRuns] = useState<string[]>([]);
    const [selectedRun, setSelectedRun] = useState<string>('');
    const [data, setData] = useState<RunData | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [activeTab, setActiveTab] = useState<'training' | 'system'>('training');

    useEffect(() => {
        if (selectedFolder) {
            setLoading(true);
            getStatsRuns(selectedFolder)
                .then((fetchedRuns) => {
                    setRuns(fetchedRuns);
                    if (fetchedRuns.length > 0) {
                        setSelectedRun(fetchedRuns[0]);
                    } else {
                        setSelectedRun('');
                        setData(null);
                    }
                })
                .catch((err) => console.error(err))
                .finally(() => setLoading(false));
        }
    }, [selectedFolder]);

    useEffect(() => {
        if (selectedFolder && selectedRun) {
            setLoading(true);
            getRunData(selectedFolder, selectedRun)
                .then((fetchedData) => {
                    setData(fetchedData);
                })
                .catch((err) => console.error(err))
                .finally(() => setLoading(false));
        }
    }, [selectedFolder, selectedRun]);

    if (folders.length === 0) {
        return <div className="text-center text-gray-500">No device folders found.</div>;
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                        <label htmlFor="folder-select" className="font-medium text-gray-700 dark:text-gray-300">
                            Device:
                        </label>
                        <select
                            id="folder-select"
                            value={selectedFolder}
                            onChange={(e) => setSelectedFolder(e.target.value)}
                            className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-800 dark:border-gray-700 dark:text-white sm:text-sm p-2 border"
                        >
                            {folders.map((folder) => (
                                <option key={folder} value={folder}>
                                    {folder}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="flex items-center space-x-2">
                        <label htmlFor="run-select" className="font-medium text-gray-700 dark:text-gray-300">
                            Run:
                        </label>
                        <select
                            id="run-select"
                            value={selectedRun}
                            onChange={(e) => setSelectedRun(e.target.value)}
                            className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-800 dark:border-gray-700 dark:text-white sm:text-sm p-2 border"
                        >
                            {runs.map((run) => (
                                <option key={run} value={run}>
                                    {run}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                <div className="flex space-x-2 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
                    <button
                        onClick={() => setActiveTab('training')}
                        className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${activeTab === 'training'
                            ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow'
                            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                            }`}
                    >
                        Training Metrics
                    </button>
                    <button
                        onClick={() => setActiveTab('system')}
                        className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${activeTab === 'system'
                            ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow'
                            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                            }`}
                    >
                        System Metrics
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="text-center py-10">Loading data...</div>
            ) : !data ? (
                <div className="text-center py-10">No data available</div>
            ) : (
                <div className="space-y-8">
                    {activeTab === 'training' && (
                        <>
                            {/* Loss Chart */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                                <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white mb-4">
                                    Training Loss
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={data.training}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
                                            <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                            />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="loss"
                                                stroke="#ef4444"
                                                activeDot={{ r: 8 }}
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Accuracy Chart */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                                <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white mb-4">
                                    Accuracy
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={data.training}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
                                            <YAxis label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                            />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="accuracy"
                                                stroke="#3b82f6"
                                                activeDot={{ r: 8 }}
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Learning Rate Chart */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                                <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white mb-4">
                                    Learning Rate
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={data.training}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }} />
                                            <YAxis label={{ value: 'LR', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                            />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="learning_rate"
                                                stroke="#10b981"
                                                activeDot={{ r: 8 }}
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </>
                    )}

                    {activeTab === 'system' && (
                        <>
                            {/* CPU & Memory Percent Chart */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                                <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white mb-4">
                                    CPU & Memory Usage (%)
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={data.system}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                dataKey="timestamp"
                                                tickFormatter={(str) => new Date(str).toLocaleTimeString()}
                                                label={{ value: 'Time', position: 'insideBottomRight', offset: -5 }}
                                            />
                                            <YAxis label={{ value: 'Percent', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                                labelFormatter={(label) => new Date(label).toLocaleString()}
                                            />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="cpu_percent"
                                                name="CPU %"
                                                stroke="#f59e0b"
                                                dot={false}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="memory_percent"
                                                name="Memory %"
                                                stroke="#8b5cf6"
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Memory Usage GB */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                                <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white mb-4">
                                    Memory Usage (GB)
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={data.system}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                dataKey="timestamp"
                                                tickFormatter={(str) => new Date(str).toLocaleTimeString()}
                                            />
                                            <YAxis label={{ value: 'GB', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                                labelFormatter={(label) => new Date(label).toLocaleString()}
                                            />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="memory_used_gb"
                                                name="Memory Used (GB)"
                                                stroke="#ec4899"
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Network Traffic */}
                            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow">
                                <h3 className="text-lg font-medium leading-6 text-gray-900 dark:text-white mb-4">
                                    Network Traffic (MB)
                                </h3>
                                <div className="h-80">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={data.system}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                dataKey="timestamp"
                                                tickFormatter={(str) => new Date(str).toLocaleTimeString()}
                                            />
                                            <YAxis label={{ value: 'MB', angle: -90, position: 'insideLeft' }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                                labelFormatter={(label) => new Date(label).toLocaleString()}
                                            />
                                            <Legend />
                                            <Line
                                                type="monotone"
                                                dataKey="net_sent_mb"
                                                name="Sent (MB)"
                                                stroke="#14b8a6"
                                                dot={false}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="net_recv_mb"
                                                name="Received (MB)"
                                                stroke="#6366f1"
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}
