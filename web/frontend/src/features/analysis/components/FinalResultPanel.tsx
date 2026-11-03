import { MemoryStick, Timer } from "lucide-react";

import type { FinalResult } from "../types";


function formatTime(value: number | null) {
  return value === null ? "Not supported" : `${value.toFixed(4)} µs`;
}


export function FinalResultPanel({ result }: { result: FinalResult }) {
  return (
    <section className="final-results" aria-label="Final result">
      <div><span className="result-icon"><Timer size={18} /></span><p>Theoretical lower bound</p><strong>{formatTime(result.theoretical_lower_bound_us)}</strong></div>
      <div><span className="result-icon warning"><MemoryStick size={18} /></span><p>Bottleneck</p><strong>{result.bottleneck ? `${result.bottleneck[0].toUpperCase()}${result.bottleneck.slice(1)} bound` : "Not supported"}</strong></div>
      <footer><span>{result.support_status ?? "unknown"}</span><span>Confidence: {result.confidence ?? "unknown"}</span></footer>
    </section>
  );
}
