import { AnalysisPage } from "../features/analysis/AnalysisPage";


export function App() {
  return (
    <div className="app-frame">
      <nav className="topbar"><div className="brand-mark">S</div><strong>SOLAR</strong><span>Offline Performance Analysis</span></nav>
      <AnalysisPage />
    </div>
  );
}
