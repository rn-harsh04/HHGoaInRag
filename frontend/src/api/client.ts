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
    const errorData = await resp.json().catch(() => null);
    const msg = errorData?.detail?.user_message || errorData?.detail || `Voice request failed (${resp.status})`;
    throw new Error(msg);
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
    const errorData = await resp.json().catch(() => null);
    const msg = errorData?.detail?.user_message || errorData?.detail || `Text request failed (${resp.status})`;
    throw new Error(msg);
  }
  return resp.json();
}
