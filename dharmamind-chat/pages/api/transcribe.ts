import type { NextApiRequest, NextApiResponse } from 'next';
import { IncomingForm, Fields, Files, File } from 'formidable';
import fs from 'fs';

// Disable default body parser for file uploads
export const config = {
    api: {
        bodyParser: false,
    },
};

interface TranscriptionResponse {
    transcript?: string;
    error?: string;
    language?: string;
}

// Parse form data with files
const parseForm = (req: NextApiRequest): Promise<{ fields: Fields; files: Files }> => {
    return new Promise((resolve, reject) => {
        const form = new IncomingForm({
            maxFileSize: 10 * 1024 * 1024, // 10MB max
            allowEmptyFiles: false,
        });

        form.parse(req, (err, fields, files) => {
            if (err) reject(err);
            else resolve({ fields, files });
        });
    });
};

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse<TranscriptionResponse>
) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { fields, files } = await parseForm(req);

        const audioFile = files.audio;
        if (!audioFile) {
            return res.status(400).json({ error: 'No audio file provided' });
        }

        // Get the first file if it's an array
        const file = Array.isArray(audioFile) ? audioFile[0] : audioFile;
        const language = Array.isArray(fields.language) ? fields.language[0] : fields.language || 'en';

        // Read the audio file
        const audioBuffer = fs.readFileSync(file.filepath);

        // Option 1: Use OpenAI Whisper API (recommended)
        if (process.env.OPENAI_API_KEY) {
            const transcript = await transcribeWithOpenAI(audioBuffer, file.originalFilename || 'audio.webm', language);

            // Clean up temp file
            fs.unlinkSync(file.filepath);

            return res.status(200).json({ transcript, language });
        }

        // Option 2: Use backend transcription service
        if (process.env.BACKEND_URL) {
            const transcript = await transcribeWithBackend(audioBuffer, file.originalFilename || 'audio.webm', language);

            // Clean up temp file
            fs.unlinkSync(file.filepath);

            return res.status(200).json({ transcript, language });
        }

        // Option 3: Use Google Cloud Speech-to-Text (if configured)
        if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
            const transcript = await transcribeWithGoogle(audioBuffer, language);

            // Clean up temp file
            fs.unlinkSync(file.filepath);

            return res.status(200).json({ transcript, language });
        }

        // Clean up temp file
        fs.unlinkSync(file.filepath);

        // No transcription service configured
        return res.status(503).json({
            error: 'Transcription service not configured. Please use Chrome, Edge, or Safari for native voice support.'
        });

    } catch (error: any) {
        console.error('Transcription error:', error);
        return res.status(500).json({ error: 'Failed to transcribe audio' });
    }
}

// OpenAI Whisper transcription
async function transcribeWithOpenAI(audioBuffer: Buffer, filename: string, language: string): Promise<string> {
    const FormData = (await import('form-data')).default;
    const formData = new FormData();

    formData.append('file', audioBuffer, {
        filename,
        contentType: filename.endsWith('.webm') ? 'audio/webm' : 'audio/mp4'
    });
    formData.append('model', 'whisper-1');
    formData.append('language', language.split('-')[0]); // 'en-US' -> 'en'
    formData.append('response_format', 'json');

    const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
            ...formData.getHeaders()
        },
        body: formData as any
    });

    if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    return data.text || '';
}

// Backend transcription service
async function transcribeWithBackend(audioBuffer: Buffer, filename: string, language: string): Promise<string> {
    const FormData = (await import('form-data')).default;
    const formData = new FormData();

    formData.append('audio', audioBuffer, {
        filename,
        contentType: filename.endsWith('.webm') ? 'audio/webm' : 'audio/mp4'
    });
    formData.append('language', language);

    const response = await fetch(`${process.env.BACKEND_URL}/api/v1/transcribe`, {
        method: 'POST',
        headers: formData.getHeaders(),
        body: formData as any
    });

    if (!response.ok) {
        throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();
    return data.transcript || data.text || '';
}

// Google Cloud Speech-to-Text (optional - requires @google-cloud/speech)
async function transcribeWithGoogle(audioBuffer: Buffer, language: string): Promise<string> {
    // Google Cloud Speech is optional - skip if not installed
    throw new Error('Google Cloud Speech not configured');
}
