import React from "react";
import type { ResponseStatus } from "../types/api";

interface StatusBadgeProps {
  status: ResponseStatus;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  switch (status) {
    case "success":
      return <span className="status status-success">✓ Grounded</span>;
    case "refusal_insufficient_info":
      return <span className="status status-refusal">⚠️ Insufficient evidence</span>;
    case "refusal_unsafe":
      return <span className="status status-unsafe">🛡️ Safety check</span>;
    case "error":
    default:
      return <span className="status status-error">✕ Error</span>;
  }
};
