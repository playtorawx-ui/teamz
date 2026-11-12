/**
 * Browser-local LLM settings. Stored in `localStorage` under its own key so a
 * workspace reset never wipes the user's API key, and nothing here is ever
 * bundled into the build or committed.
 */

export type Provider = 'orchestrator' | 'pollinations' | 'custom' | 'mock';

export interface LlmSettings {
  provider: Provider;
  /** Empty → the tier's default model. */
  model: string;
  /** Optional Pollinations API key (`sk_…`). Empty → anonymous tier. */
  apiKey: string;
  /** Optional referrer / app name sent to Pollinations. */
  referrer: string;
  /** Stream tokens as they arrive. */
  stream: boolean;
  /** Base URL of the teamz FastAPI orchestrator (provider === 'orchestrator'). */
  orchestratorUrl: string;
}

export const SETTINGS_KEY = 'teamz:llm-settings:v1';

const envMode = (import.meta.env?.VITE_API_MODE as string | undefined) ?? '';

/** `VITE_API_MODE=live` keeps the old custom-endpoint behaviour as the default. */
export const DEFAULT_PROVIDER: Provider =
  envMode === 'live' ? 'custom' : envMode === 'mock' ? 'mock' : 'pollinations';

export const DEFAULT_SETTINGS: LlmSettings = {
  provider: DEFAULT_PROVIDER,
  model: '',
  apiKey: '',
  referrer: 'teamz',
  stream: true,
  orchestratorUrl: 'http://localhost:8000',
};

export function normaliseSettings(raw: unknown): LlmSettings {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<LlmSettings>;
  const provider: Provider =
    r.provider === 'custom' ||
    r.provider === 'mock' ||
    r.provider === 'pollinations' ||
    r.provider === 'orchestrator'
      ? r.provider
      : DEFAULT_SETTINGS.provider;
  return {
    provider,
    model: typeof r.model === 'string' ? r.model : DEFAULT_SETTINGS.model,
    apiKey: typeof r.apiKey === 'string' ? r.apiKey : '',
    referrer: typeof r.referrer === 'string' ? r.referrer : DEFAULT_SETTINGS.referrer,
    stream: typeof r.stream === 'boolean' ? r.stream : DEFAULT_SETTINGS.stream,
    orchestratorUrl:
      typeof r.orchestratorUrl === 'string' && r.orchestratorUrl.trim()
        ? r.orchestratorUrl.trim().replace(/\/$/, '')
        : DEFAULT_SETTINGS.orchestratorUrl,
  };
}

export function loadSettings(): LlmSettings {
  try {
    const raw = globalThis.localStorage?.getItem(SETTINGS_KEY);
    return normaliseSettings(raw ? JSON.parse(raw) : {});
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

type Listener = (s: LlmSettings) => void;
const listeners = new Set<Listener>();
let current: LlmSettings | null = null;

export function getSettings(): LlmSettings {
  if (!current) current = loadSettings();
  return current;
}

export function saveSettings(next: LlmSettings): void {
  current = normaliseSettings(next);
  try {
    globalThis.localStorage?.setItem(SETTINGS_KEY, JSON.stringify(current));
  } catch {
    /* storage disabled — settings still apply for this session */
  }
  listeners.forEach((l) => l(current!));
}

export function subscribeSettings(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
