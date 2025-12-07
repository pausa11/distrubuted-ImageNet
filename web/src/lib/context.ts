import fs from 'fs/promises';
import path from 'path';

const REPO_ROOT = path.join(process.cwd(), '..'); // Assuming web is at root/web
const NEURAL_NETWORK_TRAIN_DIR = path.join(REPO_ROOT, 'neural-network-train');

async function getFilesRecursively(dir: string): Promise<string[]> {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    const files = await Promise.all(entries.map(async (entry) => {
        const res = path.resolve(dir, entry.name);
        return entry.isDirectory() ? getFilesRecursively(res) : res;
    }));
    return Array.prototype.concat(...files);
}

export async function getCodebaseContext(): Promise<string> {
    try {
        const srcDir = path.join(NEURAL_NETWORK_TRAIN_DIR, 'src');
        const scriptsDir = path.join(NEURAL_NETWORK_TRAIN_DIR, 'scripts');

        const srcFiles = await getFilesRecursively(srcDir);
        const scriptsFiles = await getFilesRecursively(scriptsDir);

        const allFiles = [...srcFiles, ...scriptsFiles].filter(file => file.endsWith('.py') || file.endsWith('.ipynb') || file.endsWith('.md'));

        let context = "Here is the context of the 'neural-network-train' codebase:\n\n";

        for (const file of allFiles) {
            const content = await fs.readFile(file, 'utf-8');
            const relativePath = path.relative(NEURAL_NETWORK_TRAIN_DIR, file);
            context += `--- START OF FILE: ${relativePath} ---\n`;
            context += content;
            context += `\n--- END OF FILE: ${relativePath} ---\n\n`;
        }

        return context;
    } catch (error) {
        console.error("Error reading codebase context:", error);
        return "Error reading codebase context.";
    }
}
