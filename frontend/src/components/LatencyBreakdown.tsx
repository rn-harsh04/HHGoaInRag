import React from "react";

interface LatencyBreakdownProps {
  timings?: {
    stt: number;
    retrieval: number;
    llm: number;
    total: number;
  };
}

export const LatencyBreakdown: React.FC<LatencyBreakdownProps> = ({ timings }) => {
  if (!timings) return null;

  const retrievalSlaMet = timings.retrieval <= 200;
  const progressPercent = Math.min(100, (timings.retrieval / 200) * 100);

  return (
    <div className="panel latency-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <span className="panel-icon">⚡</span>
          <h3>Latency Analytics & SLA Compliance</h3>
        </div>
        <span
          className={`sla-badge ${
            retrievalSlaMet ? "sla-badge-pass" : "sla-badge-fail"
          }`}
        >
          {retrievalSlaMet
            ? `✓ SLA Passed (${timings.retrieval.toFixed(1)}ms ≤ 200ms)`
            : `⚠️ SLA Missed (${timings.retrieval.toFixed(1)}ms > 200ms)`}
        </span>
      </div>

      <div className="sla-progress-container">
        <div className="sla-progress-labels">
          <span>0 ms</span>
          <span className="sla-marker-label">Target: 200 ms SLA</span>
          <span>&gt;200 ms</span>
        </div>
        <div className="sla-progress-track">
          <div
            className={`sla-progress-bar ${
              retrievalSlaMet ? "bar-pass" : "bar-fail"
            }`}
            style={{ width: `${progressPercent}%` }}
          />
          <div className="sla-target-line" style={{ left: "100%" }} />
        </div>
      </div>

      <div className="timings-grid">
        <div className="timing-card">
          <span className="timing-label">🎙️ STT (Sarvam)</span>
          <span className="timing-value">{timings.stt.toFixed(0)} ms</span>
        </div>
        <div className="timing-card highlight-card">
          <span className="timing-label">🔍 Hybrid Retrieval</span>
          <span className="timing-value">{timings.retrieval.toFixed(1)} ms</span>
          <span className="timing-sub">FAISS + BM25 + RRF</span>
        </div>
        <div className="timing-card">
          <span className="timing-label">🤖 LLM Generation</span>
          <span className="timing-value">{timings.llm.toFixed(0)} ms</span>
        </div>
        <div className="timing-card total-card">
          <span className="timing-label">⏱️ Total Pipeline</span>
          <span className="timing-value">{timings.total.toFixed(0)} ms</span>
        </div>
      </div>
    </div>
  );
};
