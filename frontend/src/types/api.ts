export type ResponseStatus =
  | "success"
  | "refusal_insufficient_info"
  | "refusal_unsafe"
  | "error";

export interface ChunkHit {
  chunk_id: string;
  strategy: string;
  text: string;
  score: number;
  parent_id: string;
}

export interface VoiceQueryResponse {
  status: ResponseStatus;
  transcript?: { text: string; language: string; stt_latency_ms: number };
  retrieval?: {
    chunks: ChunkHit[];
    max_score: number;
    latency_ms: number;
  };
  answer?: {
    text: string;
    confidence: number;
    used_chunk_ids: string[];
    refused: boolean;
    refusal_reason?: string | null;
  };
  groundedness?: { score: number; passed: boolean };
  timings_ms?: {
    stt: number;
    retrieval: number;
    llm: number;
    total: number;
  };
  request_id: string;
  message?: string | null;
}
