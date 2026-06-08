import { Settings } from "lucide-react";
import type { ReactElement } from "react";
import { useEffect, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ModelStatusBar from "./components/ModelStatusBar";
import QueryBar from "./components/QueryBar";
import SettingsPanel from "./components/SettingsPanel";
import { useChatStore } from "./stores/chatStore";

export default function App(): ReactElement {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const setStatus = useChatStore((state) => state.setStatus);

  useEffect(() => {
    const controller = new AbortController();
    const pollStatus = async (): Promise<void> => {
      try {
        const response = await fetch("/api/status", { signal: controller.signal });
        if (response.ok) {
          setStatus(await response.json());
        }
      } catch {
        if (!controller.signal.aborted) {
          setStatus({
            cpu_percent: 0,
            ram_mb: 0,
            tokens_per_sec: 0,
            model_loaded: false
          });
        }
      }
    };
    pollStatus();
    const interval = window.setInterval(pollStatus, 2000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [setStatus]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CPU-native SLM</p>
          <h1>CogniCore</h1>
        </div>
        <button
          className="icon-button"
          type="button"
          aria-label="Open settings"
          title="Settings"
          onClick={() => setSettingsOpen(true)}
        >
          <Settings size={20} />
        </button>
      </header>
      <ModelStatusBar />
      <ChatWindow />
      <QueryBar />
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </main>
  );
}
