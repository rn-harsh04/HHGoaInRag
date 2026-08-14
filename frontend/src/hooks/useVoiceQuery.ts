import { useState, useRef, useCallback, useEffect } from "react";
import { submitVoiceQuery, submitTextQuery } from "../api/client";
import type { VoiceQueryResponse } from "../types/api";

export type RecordingState = "idle" | "recording" | "processing";

export interface UseVoiceQueryReturn {
  state: RecordingState;
  recordingDuration: number;
  response: VoiceQueryResponse | null;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  submitText: (text: string) => Promise<void>;
  reset: () => void;
}

export function useVoiceQuery(languageHint = "en-IN"): UseVoiceQueryReturn {
  const [state, setState] = useState<RecordingState>("idle");
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [response, setResponse] = useState<VoiceQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const handleAudioAvailable = useCallback(
    async (blob: Blob) => {
      setState("processing");
      setError(null);
      try {
        const res = await submitVoiceQuery(blob, languageHint);
        setResponse(res);
      } catch (err: any) {
        setError(err.message || "Failed to process voice query");
      } finally {
        setState("idle");
      }
    },
    [languageHint]
  );

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setResponse(null);
      audioChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        handleAudioAvailable(audioBlob);
      };

      mediaRecorder.start(250);
      setState("recording");
      setRecordingDuration(0);

      const startTime = Date.now();
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        setRecordingDuration(elapsed);

        // Max 25s auto-stop safeguard
        if (elapsed >= 25) {
          stopRecording();
        }
      }, 200);
    } catch (err: any) {
      setError(err.message || "Could not access microphone");
      setState("idle");
    }
  }, [handleAudioAvailable, stopRecording]);

  const submitText = useCallback(async (text: string) => {
    if (!text.trim()) return;
    setState("processing");
    setError(null);
    try {
      const res = await submitTextQuery(text);
      setResponse(res);
    } catch (err: any) {
      setError(err.message || "Failed to process text query");
    } finally {
      setState("idle");
    }
  }, []);

  const reset = useCallback(() => {
    stopRecording();
    setState("idle");
    setResponse(null);
    setError(null);
    setRecordingDuration(0);
  }, [stopRecording]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  return {
    state,
    recordingDuration,
    response,
    error,
    startRecording,
    stopRecording,
    submitText,
    reset,
  };
}
