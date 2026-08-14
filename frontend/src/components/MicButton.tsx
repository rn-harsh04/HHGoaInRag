import React from "react";
import type { RecordingState } from "../hooks/useVoiceQuery";

interface MicButtonProps {
  state: RecordingState;
  duration: number;
  onStart: () => void;
  onStop: () => void;
  selectedLang: string;
  onLangChange: (lang: string) => void;
}

export const MicButton: React.FC<MicButtonProps> = ({
  state,
  duration,
  onStart,
  onStop,
  selectedLang,
  onLangChange,
}) => {
  const isRecording = state === "recording";
  const isProcessing = state === "processing";

  return (
    <div className="mic-container">
      <div className="mic-controls-row">
        <button
          type="button"
          className={`mic-button ${isRecording ? "recording" : ""} ${
            isProcessing ? "processing" : ""
          }`}
          onClick={isRecording ? onStop : onStart}
          disabled={isProcessing}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
        >
          <div className="mic-icon-wrapper">
            {isRecording ? (
              <span className="icon-stop">■</span>
            ) : isProcessing ? (
              <span className="spinner" />
            ) : (
              <svg
                className="mic-svg"
                viewBox="0 0 24 24"
                width="28"
                height="28"
                fill="currentColor"
              >
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )}
          </div>
          <span className="mic-label">
            {isRecording
              ? `Recording... 0:${duration.toString().padStart(2, "0")}`
              : isProcessing
              ? "Transcribing & Retrieving..."
              : "Speak Question"}
          </span>
        </button>

        <div className="lang-picker">
          <label htmlFor="language-select">Voice Model</label>
          <select
            id="language-select"
            value={selectedLang}
            onChange={(e) => onLangChange(e.target.value)}
            disabled={isRecording || isProcessing}
          >
            <option value="en-IN">English (India) - Sarvam</option>
            <option value="hi-IN">Hindi (hi-IN)</option>
            <option value="ta-IN">Tamil (ta-IN)</option>
            <option value="bn-IN">Bengali (bn-IN)</option>
          </select>
        </div>
      </div>
      {isRecording && (
        <div className="recording-waveform">
          <span className="bar b1" />
          <span className="bar b2" />
          <span className="bar b3" />
          <span className="bar b4" />
          <span className="bar b5" />
          <span className="time-limit-hint">Max 25s auto-limit</span>
        </div>
      )}
    </div>
  );
};
