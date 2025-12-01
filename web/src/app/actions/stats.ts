'use server';

import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const STATS_DIR = '/home/daniel/distrubuted-ImageNet/neural-network-train/src/stats/kali';

export interface TrainingMetric {
    timestamp: string;
    epoch: number;
    batch: number;
    loss: number | null;
    accuracy: number | null;
    learning_rate: number | null;
}

export interface SystemMetric {
    timestamp: string;
    cpu_percent: number;
    memory_percent: number;
    memory_used_gb: number;
    net_sent_mb: number;
    net_recv_mb: number;
}

export interface RunData {
    training: TrainingMetric[];
    system: SystemMetric[];
}

export async function getStatsRuns(): Promise<string[]> {
    try {
        const files = await fs.promises.readdir(STATS_DIR);
        const runs = new Set<string>();

        files.forEach((file) => {
            // Match run_X_...
            const match = file.match(/^(run_\d+)_/);
            if (match) {
                runs.add(match[1]);
            } else if (file === 'training_metrics.csv' || file === 'system_metrics.csv') {
                runs.add('global');
            }
        });

        return Array.from(runs).sort((a, b) => {
            if (a === 'global') return -1;
            if (b === 'global') return 1;
            // Sort by number
            const numA = parseInt(a.replace('run_', ''));
            const numB = parseInt(b.replace('run_', ''));
            return numA - numB;
        });
    } catch (error) {
        console.error('Error reading stats directory:', error);
        return [];
    }
}

async function parseCsv<T>(filename: string): Promise<T[]> {
    try {
        const filePath = path.join(STATS_DIR, filename);
        // Check if file exists
        try {
            await fs.promises.access(filePath);
        } catch {
            return [];
        }

        const fileContent = await fs.promises.readFile(filePath, 'utf-8');
        const { data } = Papa.parse<T>(fileContent, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            transformHeader: (header) => header.trim(),
        });
        return data;
    } catch (error) {
        console.error(`Error reading file ${filename}:`, error);
        return [];
    }
}

export async function getRunData(runId: string): Promise<RunData> {
    const trainingFile = runId === 'global' ? 'training_metrics.csv' : `${runId}_training_metrics.csv`;
    const systemFile = runId === 'global' ? 'system_metrics.csv' : `${runId}_system_metrics.csv`;

    const [training, system] = await Promise.all([
        parseCsv<TrainingMetric>(trainingFile),
        parseCsv<SystemMetric>(systemFile)
    ]);

    return { training, system };
}
