import type { VoiceQueryResponse } from "../types/api";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

export async function submitVoiceQuery(
  blob: Blob,
  languageHint: string
): Promise<VoiceQueryResponse> {
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  form.append("language_hint", languageHint);

  const resp = await fetch(`${API_BASE}/v1/query/voice`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return resp.json();
}

export async function submitTextQuery(query: string): Promise<VoiceQueryResponse> {
  const resp = await fetch(`${API_BASE}/v1/query/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, language: "en" }),
  });
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return resp.json();
}
