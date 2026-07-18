import { useEffect, useState } from "react";
import App from "./App";
import { ObservabilityDashboard } from "./ObservabilityDashboard";
import "./observability.css";

export default function Root() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const update = () => setHash(window.location.hash);
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);

  if (hash === "#observability") return <ObservabilityDashboard />;
  return (
    <>
      <App />
      <a className="telemetry-launch" href="#observability">
        <span>●</span> AGENT 可观测看板
      </a>
    </>
  );
}
