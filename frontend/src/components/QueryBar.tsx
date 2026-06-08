import { Search, Send } from "lucide-react";
import type { FormEvent, ReactElement } from "react";
import { useState } from "react";
import { useChatStore } from "../stores/chatStore";

export default function QueryBar(): ReactElement {
  const [value, setValue] = useState("");
  const addMessage = useChatStore((state) => state.addMessage);
  const appendToMessage = useChatStore((state) => state.appendToMessage);
  const finishMessage = useChatStore((state) => state.finishMessage);
  const setGenerating = useChatStore((state) => state.setGenerating);
  const isGenerating = useChatStore((state) => state.isGenerating);
  const useWebSearch = useChatStore((state) => state.useWebSearch);
  const maxTokens = useChatStore((state) => state.maxTokens);
  const setUseWebSearch = useChatStore((state) => state.setUseWebSearch);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const message = value.trim();
    if (!message || isGenerating) {
      return;
    }
    const assistantId = crypto.randomUUID();
    addMessage({
      id: crypto.randomUUID(),
      sender: "user",
      content: message,
      createdAt: new Date().toISOString(),
      streaming: false
    });
    addMessage({
      id: assistantId,
      sender: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
      streaming: true
    });
    setValue("");
    setGenerating(true);

    await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, use_web_search: useWebSearch, max_tokens: maxTokens })
    });
    const events = new EventSource("/api/stream");
    events.onmessage = (streamEvent: MessageEvent<string>): void => {
      const payload = JSON.parse(streamEvent.data) as { token: string; done: boolean };
      if (payload.done) {
        finishMessage(assistantId);
        setGenerating(false);
        events.close();
        return;
      }
      appendToMessage(assistantId, payload.token);
    };
    events.onerror = (): void => {
      appendToMessage(assistantId, "\n\nConnection interrupted.");
      finishMessage(assistantId);
      setGenerating(false);
      events.close();
    };
  };

  return (
    <form className="query-bar" onSubmit={submit}>
      <button
        className={`toggle-button ${useWebSearch ? "active" : ""}`}
        type="button"
        aria-label="Toggle web search"
        title="Web search"
        onClick={() => setUseWebSearch(!useWebSearch)}
      >
        <Search size={18} />
      </button>
      <textarea
        value={value}
        rows={1}
        placeholder="Ask CogniCore..."
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      <button
        className="send-button"
        type="submit"
        aria-label="Send message"
        title="Send"
        disabled={isGenerating || value.trim().length === 0}
      >
        <Send size={19} />
      </button>
    </form>
  );
}
