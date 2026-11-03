import { useEffect } from "react";

import type { RunSnapshot } from "./types";


export function useRunEvents(
  runId: string | null,
  onSnapshot: (snapshot: RunSnapshot) => void,
  onError: () => void,
) {
  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(`/api/v1/runs/${runId}/events`);
    const handleSnapshot = (event: Event) => {
      const message = event as MessageEvent<string>;
      onSnapshot(JSON.parse(message.data) as RunSnapshot);
    };
    source.addEventListener("snapshot", handleSnapshot);
    source.onerror = onError;
    return () => {
      source.removeEventListener("snapshot", handleSnapshot);
      source.close();
    };
  }, [runId, onSnapshot, onError]);
}
