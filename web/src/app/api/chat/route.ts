import { google } from '@ai-sdk/google';
import { streamText, convertToCoreMessages } from 'ai';
import { getCodebaseContext } from '@/lib/context';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
    try {
        const { messages } = await req.json();
        const context = await getCodebaseContext();

        const result = streamText({
            model: google('gemini-2.0-flash'),
            system: `You are a helpful assistant for the 'neural-network-train' project.
        You have access to the codebase context below.
        Answer questions based on this context.
        If the user asks about the code, explain it clearly.
        
        ${context}`,
            messages: convertToCoreMessages(messages),
        });

        return result.toUIMessageStreamResponse();
    } catch (error) {
        console.error("API Route Error:", error);
        return new Response(JSON.stringify({ error: "Internal Server Error", details: error instanceof Error ? error.message : String(error) }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
}
