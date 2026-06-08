import { Cpu, Gauge, HardDrive, RadioTower } from "lucide-react";
import type { ReactElement } from "react";
import { useChatStore } from "../stores/chatStore";

export default function ModelStatusBar(): ReactElement {
  const status = useChatStore((state) => state.status);
  return (
    <section className="status-bar" aria-label="Model status">
      <div className="status-pill" title="CPU usage">
        <Cpu size={16} />
        <span>{status.cpu_percent.toFixed(1)}%</span>
      </div>
      <div className="status-pill" title="Memory usage">
        <HardDrive size={16} />
        <span>{status.ram_mb.toFixed(0)} MB</span>
      </div>
      <div className="status-pill" title="Generation speed">
        <Gauge size={16} />
        <span>{status.tokens_per_sec.toFixed(1)} tok/s</span>
      </div>
      <div className={`status-pill ${status.model_loaded ? "ready" : "offline"}`} title="Model loaded">
        <RadioTower size={16} />
        <span>{status.model_loaded ? "loaded" : "dev stream"}</span>
      </div>
    </section>
  );
}
