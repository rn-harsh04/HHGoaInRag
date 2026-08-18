import React, { useState } from "react";
import { useVoiceQuery } from "./hooks/useVoiceQuery";
import { MicButton } from "./components/MicButton";
import { StatusBadge } from "./components/StatusBadge";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { ChunksPanel } from "./components/ChunksPanel";
import { AnswerPanel } from "./components/AnswerPanel";
import { LatencyBreakdown } from "./components/LatencyBreakdown";
import titleImage from "../title.png";

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
    <div className="hh-app-shell">
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />

      <header className="topbar">
        <a className="brand-wrap" href="#voice" aria-label="Hack House Goa 2026">
          <img className="brand-title" src={titleImage} alt="Hack House Goa" />
        </a>

        <nav className="topnav" aria-label="Main navigation">
          <a className="voice-heard-link" href="#voice">Get your voice heard</a>
        </nav>
      </header>

      <main className="hero-layout">
        <section className="hero-copy">
          <span className="eyebrow">VOICE-ENABLED RAG</span>
          <h1>Ask in your voice. Get grounded answers.</h1>
          <p>
            A cleaner, faster, more human way to explore knowledge — designed for the HH Goa build culture,
            with real-time voice capture and context-aware responses.
          </p>

          <div className="stat-row">
            <div className="stat-card">
              <strong>Voice-first</strong>
              <span>hands-free input</span>
            </div>
            <div className="stat-card">
              <strong>Grounded</strong>
              <span>evidence-backed output</span>
            </div>
            <div className="stat-card">
              <strong>Fast</strong>
              <span>low-latency workflow</span>
            </div>
          </div>
        </section>

        <section className="voice-console" id="voice">
          <div className="console-header">
            <span className="console-title">Live Studio</span>
            <span className="console-pill">Query ready</span>
          </div>

          <MicButton
            state={state}
            duration={recordingDuration}
            onStart={startRecording}
            onStop={stopRecording}
            selectedLang={selectedLang}
            onLangChange={setSelectedLang}
          />

          <div className="divider-line">
            <span>or type</span>
          </div>

          <form onSubmit={handleTextSubmit} className="text-query-form">
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Ask anything..."
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
        </section>
      </main>

      {error && (
        <div className="alert-banner">
          <span className="alert-icon">!</span>
          <div className="alert-content">
            <strong>Error:</strong> {error}
          </div>
          <button className="btn-close" onClick={reset}>
            ×
          </button>
        </div>
      )}

      {response && (
        <section className="chat-panel" id="workspace">
          <div className="response-header">
            <h2>Latest response</h2>
            <div className="response-actions">
              <StatusBadge status={response.status} />
              <button className="btn-reset" onClick={reset}>
                Clear
              </button>
            </div>
          </div>

          <div className="conversation-stack">
            <TranscriptPanel transcript={response.transcript} />
            <AnswerPanel
              status={response.status}
              answer={response.answer}
              groundedness={response.groundedness}
              message={response.message}
            />
            <LatencyBreakdown timings={response.timings_ms} />
            <ChunksPanel
              chunks={response.retrieval?.chunks}
              maxScore={response.retrieval?.max_score}
              latencyMs={response.retrieval?.latency_ms}
              usedChunkIds={response.answer?.used_chunk_ids}
            />
          </div>
        </section>
      )}

      <footer className="app-footer">
        <p>HH Goa 2026 • Voice-enabled discovery experience</p>
      </footer>
    </div>
  );
};

export default App;
