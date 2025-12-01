'use server';

import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';
import { v4 as uuidv4 } from 'uuid';

const execAsync = promisify(exec);

const PREDICT_SCRIPT = '/home/daniel/distrubuted-ImageNet/neural-network-train/scripts/predict.py';
const CHECKPOINT_PATH = '/home/daniel/distrubuted-ImageNet/neural-network-train/checkpoints/best_checkpoint.pt';

export interface PredictionResult {
    class_index: number;
    synset: string;
    confidence: number;
    error?: string;
}

export async function predictImage(formData: FormData): Promise<PredictionResult> {
    const file = formData.get('image') as File;
    if (!file) {
        return { class_index: -1, synset: '', confidence: 0, error: 'No image provided' };
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const tempFilePath = join(tmpdir(), `upload-${uuidv4()}.jpg`);

    try {
        await writeFile(tempFilePath, buffer);

        const command = `/home/daniel/distrubuted-ImageNet/.venv/bin/python ${PREDICT_SCRIPT} --image_path "${tempFilePath}" --checkpoint_path "${CHECKPOINT_PATH}"`;

        const { stdout, stderr } = await execAsync(command);

        if (stderr) {
            // Some warnings might be printed to stderr, so we don't treat all stderr as fatal,
            // but if stdout is empty, it's likely an error.
            console.warn('Inference stderr:', stderr);
        }

        try {
            const result = JSON.parse(stdout);
            return result;
        } catch (parseError) {
            console.error('Failed to parse inference output:', stdout);
            return { class_index: -1, synset: '', confidence: 0, error: 'Failed to parse inference result' };
        }

    } catch (error) {
        console.error('Inference error:', error);
        return { class_index: -1, synset: '', confidence: 0, error: 'Internal server error during inference' };
    } finally {
        // Cleanup
        try {
            await unlink(tempFilePath);
        } catch (e) {
            console.error('Failed to delete temp file:', e);
        }
    }
}
