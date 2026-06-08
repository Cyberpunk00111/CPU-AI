import { create } from "zustand";

export type Sender = "user" | "assistant";

export interface ChatMessage {
  id: string;
  sender: Sender;
  content: string;
  createdAt: string;
  streaming: boolean;
}

export interface ModelStatus {
  cpu_percent: number;
  ram_mb: number;
  tokens_per_sec: number;
  model_loaded: boolean;
}

interface ChatState {
  messages: ChatMessage[];
  useWebSearch: boolean;
  maxTokens: number;
  modelVariant: string;
  isGenerating: boolean;
  status: ModelStatus;
  addMessage: (message: ChatMessage) => void;
  appendToMessage: (id: string, token: string) => void;
  finishMessage: (id: string) => void;
  setUseWebSearch: (value: boolean) => void;
  setMaxTokens: (value: number) => void;
  setModelVariant: (value: string) => void;
  setGenerating: (value: boolean) => void;
  setStatus: (status: ModelStatus) => void;
}

const initialStatus: ModelStatus = {
  cpu_percent: 0,
  ram_mb: 0,
  tokens_per_sec: 0,
  model_loaded: false
};

export const useChatStore = create<ChatState>((set) => ({
  messages: JSON.parse(window.localStorage.getItem("cognicore.messages") ?? "[]") as ChatMessage[],
  useWebSearch: false,
  maxTokens: 128,
  modelVariant: "10M",
  isGenerating: false,
  status: initialStatus,
  addMessage: (message: ChatMessage): void =>
    set((state) => {
      const messages = [...state.messages, message];
      window.localStorage.setItem("cognicore.messages", JSON.stringify(messages));
      return { messages };
    }),
  appendToMessage: (id: string, token: string): void =>
    set((state) => {
      const messages = state.messages.map((message) =>
        message.id === id ? { ...message, content: message.content + token } : message
      );
      window.localStorage.setItem("cognicore.messages", JSON.stringify(messages));
      return { messages };
    }),
  finishMessage: (id: string): void =>
    set((state) => {
      const messages = state.messages.map((message) =>
        message.id === id ? { ...message, streaming: false } : message
      );
      window.localStorage.setItem("cognicore.messages", JSON.stringify(messages));
      return { messages };
    }),
  setUseWebSearch: (value: boolean): void => set({ useWebSearch: value }),
  setMaxTokens: (value: number): void => set({ maxTokens: value }),
  setModelVariant: (value: string): void => set({ modelVariant: value }),
  setGenerating: (value: boolean): void => set({ isGenerating: value }),
  setStatus: (status: ModelStatus): void => set({ status })
}));
