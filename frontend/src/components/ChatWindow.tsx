import type { ReactElement } from "react";
import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import { useChatStore } from "../stores/chatStore";

export default function ChatWindow(): ReactElement {
  const messages = useChatStore((state) => state.messages);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <section className="chat-window" aria-live="polite">
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="avatar-pulse" />
          <p>Ask CogniCore anything you want this CPU-native assistant to handle.</p>
        </div>
      ) : (
        messages.map((message) => <MessageBubble key={message.id} message={message} />)
      )}
      <div ref={endRef} />
    </section>
  );
}
