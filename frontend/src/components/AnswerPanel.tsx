import React from "react";
import type { ResponseStatus } from "../types/api";

interface AnswerPanelProps {
  status: ResponseStatus;
  answer?: {
    text: string;
    confidence: number;
    used_chunk_ids: string[];
    refused: boolean;
    refusal_reason?: string | null;
  };
  groundedness?: {
    score: number;
    passed: boolean;
  };
  message?: string | null;
}

export const AnswerPanel: React.FC<AnswerPanelProps> = ({
  status,
  answer,
  groundedness,
  message,
}) => {
  const isRefusal =
    status === "refusal_insufficient_info" ||
    status === "refusal_unsafe" ||
    answer?.refused;

  const isError = status === "error";

  return (
    <div
      className={`panel answer-panel ${
        isRefusal ? "panel-refusal" : isError ? "panel-error" : "panel-success"
      }`}
    >
      <div className="panel-header">
        <div className="panel-title-group">
          <span className="panel-icon">
            {isRefusal ? "⚠️" : isError ? "✕" : "🤖"}
          </span>
          <h3>
            {isRefusal
              ? "System Refusal Guardrail"
              : isError
              ? "Execution Error"
              : "Grounded Answer (Gemini 3.1 Flash Lite)"}
          </h3>
        </div>

        {!isRefusal && !isError && (
          <div className="badge-group">
            {answer?.confidence !== undefined && (
              <span className="confidence-tag">
                Confidence: {(answer.confidence * 100).toFixed(0)}%
              </span>
            )}
            {groundedness && (
              <span
                className={`grounded-tag ${
                  groundedness.passed ? "passed" : "failed"
                }`}
              >
                Overlap: {(groundedness.score * 100).toFixed(0)}%{" "}
                {groundedness.passed ? "✓" : "✗"}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="answer-body">
        {isRefusal ? (
          <div className="refusal-box">
            <p className="refusal-title">
              {status === "refusal_unsafe"
                ? "Input flagged by Safety Guardrail"
                : "Passage Evidence Insufficient"}
            </p>
            <p className="refusal-text">
              {answer?.refusal_reason ||
                answer?.text ||
                message ||
                "I don't have enough information in the indexed passages to answer this question."}
            </p>
          </div>
        ) : isError ? (
          <div className="error-box">
            <p className="error-text">
              {message || "An unexpected error occurred during processing."}
            </p>
          </div>
        ) : (
          <div className="success-answer">
            <p className="answer-text">{answer?.text}</p>
            {answer?.used_chunk_ids && answer.used_chunk_ids.length > 0 && (
              <div className="citations-row">
                <span className="citations-label">Citations:</span>
                {answer.used_chunk_ids.map((id, index) => (
                  <span key={id} className="citation-pill" title={id}>
                    [{index + 1}] {id.slice(0, 8)}...
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
