import React from "react";

interface TranscriptPanelProps {
  transcript?: {
    text: string;
    language: string;
    stt_latency_ms: number;
  };
}

export const TranscriptPanel: React.FC<TranscriptPanelProps> = ({ transcript }) => {
  if (!transcript || !transcript.text) return null;

  return (
    <div className="panel transcript-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <span className="panel-icon">🎙️</span>
          <h3>Voice capture</h3>
        </div>
        <div className="badge-group">
          <span className="lang-tag">{transcript.language}</span>
          <span className="latency-tag">{transcript.stt_latency_ms.toFixed(0)} ms</span>
        </div>
      </div>
      <div className="transcript-body">
        <p className="transcript-text">“{transcript.text}”</p>
      </div>
    </div>
  );
};
