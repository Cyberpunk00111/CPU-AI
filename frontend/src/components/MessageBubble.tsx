import { Check, Copy } from "lucide-react";
import type { ReactElement } from "react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../stores/chatStore";

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps): ReactElement {
  const [copied, setCopied] = useState(false);
  const timestamp = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(message.createdAt));

  const copyMessage = async (): Promise<void> => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  return (
    <article className={`message-row ${message.sender}`}>
      <div className={`message-bubble ${message.sender}`}>
        <div className="message-meta">
          <span>{message.sender === "user" ? "You" : "CogniCore"}</span>
          <span>{timestamp}</span>
        </div>
        <div className="message-content">
          <ReactMarkdown>{message.content || " "}</ReactMarkdown>
          {message.streaming ? <span className="cursor" /> : null}
        </div>
        <button
          className="copy-button"
          type="button"
          aria-label="Copy message"
          title="Copy"
          onClick={copyMessage}
        >
          {copied ? <Check size={15} /> : <Copy size={15} />}
        </button>
      </div>
    </article>
  );
}
