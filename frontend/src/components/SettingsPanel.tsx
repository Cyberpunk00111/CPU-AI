import { X } from "lucide-react";
import type { ReactElement } from "react";
import { useChatStore } from "../stores/chatStore";

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

export default function SettingsPanel({ open, onClose }: SettingsPanelProps): ReactElement {
  const maxTokens = useChatStore((state) => state.maxTokens);
  const setMaxTokens = useChatStore((state) => state.setMaxTokens);
  const modelVariant = useChatStore((state) => state.modelVariant);
  const setModelVariant = useChatStore((state) => state.setModelVariant);

  return (
    <aside className={`settings-panel ${open ? "open" : ""}`} aria-hidden={!open}>
      <div className="settings-header">
        <h2>Settings</h2>
        <button className="icon-button" type="button" aria-label="Close settings" title="Close" onClick={onClose}>
          <X size={20} />
        </button>
      </div>
      <label className="field-label" htmlFor="variant">
        Model
      </label>
      <select id="variant" value={modelVariant} onChange={(event) => setModelVariant(event.target.value)}>
        <option value="10M">10M Phase 1</option>
        <option value="100M">100M Phase 2</option>
      </select>
      <label className="field-label" htmlFor="maxTokens">
        Max tokens: {maxTokens}
      </label>
      <input
        id="maxTokens"
        type="range"
        min="32"
        max="512"
        step="16"
        value={maxTokens}
        onChange={(event) => setMaxTokens(Number(event.target.value))}
      />
    </aside>
  );
}
