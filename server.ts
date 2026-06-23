import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

// Initialize the Google Gemini GenAI SDK
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Set limits to support base64 image data up to 15MB
  app.use(express.json({ limit: "15mb" }));
  app.use(express.urlencoded({ limit: "15mb", extended: true }));

  // API Route: Classify waste from image base64 data
  app.post("/api/classify", async (req, res) => {
    try {
      const { imageBase64 } = req.body;

      if (!imageBase64) {
        return res.status(400).json({ error: "Missing imageBase64 parameter in request body." });
      }

      if (!process.env.GEMINI_API_KEY) {
        return res.status(500).json({
          error: "GEMINI_API_KEY is not configured on the server. Please check your developer environment settings."
        });
      }

      const promptText = `You are a waste classification AI. Analyze this image and respond ONLY with a valid JSON object, no markdown, no explanation.

Classify the waste into exactly one of these six categories:
- "Wet/Organic"
- "Dry/Recyclable"
- "Hazardous"
- "E-Waste"
- "Medical/Biomedical"
- "Non-Recyclable"

Respond with this exact JSON format:
{
  "category": "<one of the six categories above>",
  "confidence": <number between 0.60 and 0.99>,
  "detected_items": ["item1", "item2"],
  "reasoning": "<one sentence explaining why you classified it this way>"
}

If the image does not contain any waste or is unclear, respond:
{
  "category": "Non-Recyclable",
  "confidence": 0.45,
  "detected_items": ["unclear image"],
  "reasoning": "Could not clearly identify waste type from this image."
}`;

      const imagePart = {
        inlineData: {
          mimeType: "image/jpeg",
          data: imageBase64,
        },
      };

      const textPart = {
        text: promptText,
      };

      const modelsToTry = [
        "gemini-2.5-flash",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview"
      ];

      let lastError: any = null;
      let aiResponse: any = null;
      let usedModel = "";

      // Helper function to call Gemini API with retry on failure (transient errors like 503/429)
      const generateWithRetry = async (modelName: string, contents: any, maxRetries = 3, initialDelay = 1000) => {
        let delay = initialDelay;
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
          try {
            return await ai.models.generateContent({
              model: modelName,
              contents,
            });
          } catch (err: any) {
            console.warn(`[Attempt ${attempt}/${maxRetries}] Model ${modelName} call failed:`, err?.message || err);
            if (attempt === maxRetries) {
              throw err;
            }
            // Double the delay for exponential backoff
            await new Promise((resolve) => setTimeout(resolve, delay));
            delay *= 2;
          }
        }
      };

      for (const modelName of modelsToTry) {
        try {
          console.log(`Attempting waste classification with model: ${modelName}`);
          aiResponse = await generateWithRetry(modelName, { parts: [imagePart, textPart] });
          usedModel = modelName;
          console.log(`Successfully classified waste using model: ${modelName}`);
          break; // Success
        } catch (err: any) {
          console.warn(`Model ${modelName} failed after all retries:`, err?.message || err);
          lastError = err;
          continue; // Try next model fallback
        }
      }

      if (!aiResponse) {
        throw new Error(
          `Our classification service is currently experiencing high demand. Please try again in a few moments. (Details: ${lastError?.message || lastError})`
        );
      }

      const rawText = aiResponse.text;
      if (!rawText) {
        throw new Error("No response content returned from Gemini AI.");
      }

      // Isolate JSON block just in case markdown block characters were included
      let sanitizedText = rawText.trim();
      if (sanitizedText.startsWith("```json")) {
        sanitizedText = sanitizedText.replace(/^```json/, "");
      } else if (sanitizedText.startsWith("```")) {
        sanitizedText = sanitizedText.replace(/^```/, "");
      }
      if (sanitizedText.endsWith("```")) {
        sanitizedText = sanitizedText.slice(0, -3);
      }
      sanitizedText = sanitizedText.trim();

      // Safely parse the response before responding
      const parsedResult = JSON.parse(sanitizedText);
      return res.json(parsedResult);
    } catch (err: any) {
      console.error("Classification error:", err);
      return res.status(500).json({
        error: err?.message || "An unexpected error occurred during classification. Please try again."
      });
    }
  });

  // Serve static files in production vs Vite middleware in development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
