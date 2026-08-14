import React, { useState } from "react";
import { useVoiceQuery } from "./hooks/useVoiceQuery";
import { MicButton } from "./components/MicButton";
import { StatusBadge } from "./components/StatusBadge";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { ChunksPanel } from "./components/ChunksPanel";
import { AnswerPanel } from "./components/AnswerPanel";
import { LatencyBreakdown } from "./components/LatencyBreakdown";

export const App: React.FC = () => {
  const [selectedLang, setSelectedLang] = useState<string>("en-IN");
  const [textInput, setTextInput] = useState<string>("");
  const [sampleQueries] = useState<string[]>([
    "What is the capital of India?",
    "How does photosynthesis work in plants?",
    "What are renewable energy sources?",
    "How do vaccines help the immune system?",
  ]);

  const {
    state,
    recordingDuration,
    response,
    error,
    startRecording,
    stopRecording,
    submitText,
    reset,
  } = useVoiceQuery(selectedLang);

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;
    submitText(textInput.trim());
  };

  const handleSampleClick = (q: string) => {
    setTextInput(q);
    submitText(q);
  };

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo">🎙️</div>
          <div>
            <h1>Voice RAG — HH Goa 2026</h1>
            <p className="subtitle">
              Multi-Strategy Chunking • Hybrid Retrieval (FAISS + BM25) • Strict &le;200ms Retrieval SLA • Grounded Gemini
            </p>
          </div>
        </div>
        <div className="header-badges">
          <span className="badge badge-sla">⚡ SLA: &le;200ms Target</span>
          <span className="badge badge-model">🤖 Gemini 3.1 Flash Lite</span>
          <span className="badge badge-stt">🗣️ Sarvam Saaras v3</span>
        </div>
      </header>

      <main className="main-content">
        {/* Interaction Section */}
        <section className="interaction-card">
          <div className="interaction-tabs">
            <span className="tab active">Voice Input</span>
            <span className="tab-hint">Press to talk or use text fallback</span>
          </div>

          <div className="interaction-body">
            <MicButton
              state={state}
              duration={recordingDuration}
              onStart={startRecording}
              onStop={stopRecording}
              selectedLang={selectedLang}
              onLangChange={setSelectedLang}
            />

            <div className="divider">
              <span>OR TYPE QUERY</span>
            </div>

            <form onSubmit={handleTextSubmit} className="text-query-form">
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Ask any question from the indexed MSMARCO-XI corpus..."
                disabled={state === "recording" || state === "processing"}
              />
              <button
                type="submit"
                className="btn-send"
                disabled={!textInput.trim() || state === "recording" || state === "processing"}
              >
                Send
              </button>
            </form>

            <div className="sample-queries">
              <span className="sample-label">Try queries:</span>
              {sampleQueries.map((q, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="sample-pill"
                  onClick={() => handleSampleClick(q)}
                  disabled={state === "recording" || state === "processing"}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Global Error Banner */}
        {error && (
          <div className="alert-banner alert-error">
            <span className="alert-icon">⚠️</span>
            <div className="alert-content">
              <strong>Error:</strong> {error}
            </div>
            <button className="btn-close" onClick={reset}>
              ✕
            </button>
          </div>
        )}

        {/* Response Dashboard */}
        {response && (
          <section className="response-dashboard">
            <div className="response-header">
              <h2>Query Results</h2>
              <div className="response-actions">
                <StatusBadge status={response.status} />
                <button className="btn-reset" onClick={reset}>
                  Clear
                </button>
              </div>
            </div>

            {/* Transcript Panel (if voice) */}
            <TranscriptPanel transcript={response.transcript} />

            {/* Main Answer Panel */}
            <AnswerPanel
              status={response.status}
              answer={response.answer}
              groundedness={response.groundedness}
              message={response.message}
            />

            {/* Latency & SLA Panel */}
            <LatencyBreakdown timings={response.timings_ms} />

            {/* Retrieved Chunks with Strategies */}
            <ChunksPanel
              chunks={response.retrieval?.chunks}
              maxScore={response.retrieval?.max_score}
              latencyMs={response.retrieval?.latency_ms}
              usedChunkIds={response.answer?.used_chunk_ids}
            />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Task 2: Voice-Enabled RAG Model • MSMARCO-XI Corpus • 4 Chunking Strategies (Fixed, Semantic, Metadata, Parent-Child) • In-Memory FAISS HNSW + BM25Okapi
        </p>
      </footer>
    </div>
  );
};

export default App;
