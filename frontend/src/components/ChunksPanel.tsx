import React, { useState } from "react";
import type { ChunkHit } from "../types/api";

interface ChunksPanelProps {
  chunks?: ChunkHit[];
  maxScore?: number;
  latencyMs?: number;
  usedChunkIds?: string[];
}

export const ChunksPanel: React.FC<ChunksPanelProps> = ({
  chunks = [],
  maxScore = 0,
  latencyMs = 0,
  usedChunkIds = [],
}) => {
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  if (!chunks || chunks.length === 0) return null;

  const getStrategyColor = (strategy: string) => {
    switch (strategy.toLowerCase()) {
      case "fixed":
        return "strategy-fixed";
      case "semantic":
        return "strategy-semantic";
      case "metadata":
        return "strategy-metadata";
      case "child":
        return "strategy-child";
      default:
        return "strategy-default";
    }
  };

  return (
    <div className="panel chunks-panel">
      <div className="panel-header">
        <div className="panel-title-group">
          <span className="panel-icon">📚</span>
          <h3>Multi-Strategy Retrieved Chunks ({chunks.length})</h3>
        </div>
        <div className="badge-group">
          <span className="score-tag">Max Similarity: {maxScore.toFixed(3)}</span>
          <span className="sla-tag">Retrieval: {latencyMs.toFixed(1)} ms</span>
        </div>
      </div>

      <div className="strategy-legend">
        <span className="legend-item"><span className="dot fixed" /> Fixed Overlap (512/128)</span>
        <span className="legend-item"><span className="dot semantic" /> Semantic Split (Cos &gt; 0.75)</span>
        <span className="legend-item"><span className="dot metadata" /> Metadata-Aware</span>
        <span className="legend-item"><span className="dot child" /> Parent-Child Window (256/64)</span>
      </div>

      <div className="chunks-list">
        {chunks.map((chunk, idx) => {
          const isCited = usedChunkIds.includes(chunk.chunk_id);
          const isExpanded = expandedChunkId === chunk.chunk_id;

          return (
            <div
              key={chunk.chunk_id}
              className={`chunk-card ${isCited ? "cited-chunk" : ""}`}
              onClick={() => setExpandedChunkId(isExpanded ? null : chunk.chunk_id)}
            >
              <div className="chunk-header-row">
                <div className="chunk-meta-left">
                  <span className="chunk-rank">#{idx + 1}</span>
                  <span className={`strategy-pill ${getStrategyColor(chunk.strategy)}`}>
                    {chunk.strategy.toUpperCase()}
                  </span>
                  {isCited && <span className="cited-badge">★ Cited by LLM</span>}
                </div>
                <div className="chunk-meta-right">
                  <span className="chunk-score">Score: {chunk.score.toFixed(4)}</span>
                  <span className="chunk-parent-id" title={chunk.parent_id}>
                    Parent: {chunk.parent_id.split(":").slice(-2).join(":")}
                  </span>
                </div>
              </div>
              <p className={`chunk-text ${isExpanded ? "expanded" : ""}`}>
                {chunk.text}
              </p>
              <div className="chunk-expand-hint">
                {isExpanded ? "Click to collapse" : "Click to view full text"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
