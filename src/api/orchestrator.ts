/**
 * ============================================================================
 *  Orchestrator transport — talks to the teamz FastAPI server.
 * ============================================================================
 *
 *  The server (`server/`) is the keyless-first LLM router: it owns provider
 *  health, the fallback chain, usage accounting and conversation memory. This
 *  module is the browser side of that contract.
 *
 *  Everything degrades gracefully: if the orchestrator is not running, the
 *  caller falls back to the direct Pollinations transport, so the app keeps
 *  working with no server at all.
 */

import type { Agent, Message } from '../types';

/** Where the orchestrator lives. Override with `VITE_ORCHESTRATOR_URL`. */
export const ORCHESTRATOR_URL: string =
  (import.meta.env?.VITE_ORCHESTRATOR_URL as string | undefined)?.replace(/\/$/, '') ||
  'http://localhost:8000';

export interface ProviderHealth {
  status: 'unknown' | 'healthy' | 'degraded' | 'dead';
  lastChecked: number | null;
  latencyMs: number | null;
  consecutiveFailures: number;
  circuitOpen: boolean;
  circuitOpenForMs: number;
  lastError: string | null;
}

export interface ProviderInfo {
  name: string;
  baseUrl: string;
  models: string[];
  rpmLimit: number;
  isLocal: boolean;
  priority: number;
  kind: 'keyless' | 'keyed' | 'local';
  costTier: number;
  hasKey: boolean;
  remainingBudget: number | null;
  health: ProviderHealth;
}

export interface RouteDecision {
  provider: string;
  model: string;
  reason: string;
  score: number;
  candidates: { provider: string; score: number; reason: string }[];
}

export interface Hop {
  provider: string;
  model: string;
  ok: boolean;
  status: number | null;
  error: string | null;
  latencyMs: number;
  retryable: boolean;
}

export interface UsageSummary {
  requests: number;
  okRequests: number;
  errorRequests: number;
  successRate: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  costUsd: number;
  avgLatencyMs: number;
  byProvider: Record<string, { requests: number; tokens: number; costUsd: number; errors: number }>;
  byModel: Record<string, { requests: number; tokens: number; costUsd: number; errors: number }>;
  byDay: Record<string, { requests: number; tokens: number; costUsd: number }>;
}

export interface OrchestratorReply {
  text: string;
  provider: string;
  model: string;
  routing: RouteDecision;
  hops: Hop[];
  latencyMs: number;
  usage: { promptTokens: number; completionTokens: number; totalTokens: number };
}

const TIMEOUT_MS = 45000;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${ORCHESTRATOR_URL}${path}`, {
      ...init,
      signal: init?.signal ?? controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`${res.status} ${res.statusText}${detail ? ` — ${detail.slice(0, 200)}` : ''}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

/** Cheap liveness probe — used to decide whether to route through the server. */
export async function orchestratorAlive(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${ORCHESTRATOR_URL}/healthz`, { signal: controller.signal });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

export function listProviders(): Promise<{ providers: ProviderInfo[]; count: number; keylessCount: number }> {
  return request('/providers');
}

export function probeProviders(): Promise<{ health: Record<string, ProviderHealth> }> {
  return request('/providers/probe', { method: 'POST' });
}

export function previewRoute(input: {
  model?: string;
  local_only?: boolean;
  keyless_only?: boolean;
  estimated_tokens?: number;
  hard_task?: boolean;
}): Promise<RouteDecision & { estimatedTokens: number; estimatedCostUsd: number }> {
  return request('/route/preview', { method: 'POST', body: JSON.stringify(input) });
}

export function getUsage(): Promise<UsageSummary> {
  return request('/usage');
}

export function resetUsage(): Promise<{ ok: boolean }> {
  return request('/usage/reset', { method: 'POST' });
}

/** Build the OpenAI-shaped message list from teamz history. */
export function toChatMessages(
  agent: Agent,
  history: Message[],
  agentsById: Map<string, Agent>,
  opts: { conversationKind: 'direct' | 'channel'; conversationName: string; topic?: string; otherAgents?: { id: string; name: string }[]; directed?: boolean },
): { role: 'system' | 'user' | 'assistant'; content: string }[] {
  const facts: string[] = [`Your name is ${agent.name}.`];
  if (agent.origin?.role) facts.push(`Your role: ${agent.origin.role}.`);
  if (agent.skills?.length) {
    const top = agent.skills
      .slice()
      .sort((a, b) => b.level - a.level)
      .slice(0, 6)
      .map((s) => s.label);
    facts.push(`Your strongest skills: ${top.join(', ')}.`);
  }
  if (opts.conversationKind === 'channel') {
    const others = (opts.otherAgents ?? []).map((a) => a.name);
    facts.push(
      `You are in a group chat channel called "${opts.conversationName}"` +
        (opts.topic ? ` (topic: ${opts.topic})` : '') +
        ` with the user` +
        (others.length ? ` and these other AI agents: ${others.join(', ')}` : '') +
        '.',
    );
    facts.push(
      'Messages from other participants are prefixed with their name in square brackets. Reply only as yourself, with no name prefix.',
    );
    if (opts.directed) facts.push('The user addressed you directly with an @mention.');
  } else {
    facts.push('You are in a private one-to-one chat with the user.');
  }
  facts.push('Keep replies concise. Markdown is supported.');

  const system = [agent.systemPrompt?.trim() || `You are ${agent.name}, a helpful AI agent.`, facts.join('\n')].join('\n\n');

  const messages: { role: 'system' | 'user' | 'assistant'; content: string }[] = [
    { role: 'system', content: system },
  ];
  for (const m of history.slice(-24)) {
    if (m.authorType === 'agent') {
      messages.push({ role: 'assistant', content: m.text });
    } else {
      const name = m.authorType === 'user' ? 'User' : agentsById.get(m.authorId ?? '')?.name ?? 'Agent';
      messages.push({ role: 'user', content: `[${name}]: ${m.text}` });
    }
  }
  return messages;
}

/** One routed completion through the orchestrator, with the full hop trace. */
export async function sendViaOrchestrator(
  agent: Agent,
  history: Message[],
  agentsById: Map<string, Agent>,
  opts: {
    conversationId: string;
    conversationName: string;
    conversationKind: 'direct' | 'channel';
    topic?: string;
    otherAgents?: { id: string; name: string }[];
    directed?: boolean;
    signal?: AbortSignal;
    onDelta?: (fullText: string) => void;
  },
): Promise<OrchestratorReply> {
  const messages = toChatMessages(agent, history, agentsById, opts);
  const started = performance.now();

  const res = await fetch(`${ORCHESTRATOR_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: opts.signal,
    body: JSON.stringify({
      messages,
      conversation_id: opts.conversationId,
      model: agent.model?.trim() || undefined,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const hops: Hop[] = body?.error?.hops ?? [];
    const trail = hops.map((h) => `${h.provider}${h.status ? ` ${h.status}` : ''}`).join(' → ');
    throw new Error(
      `Every provider failed${trail ? ` (${trail})` : ''}. Check the orchestrator's /providers health.`,
    );
  }

  const body = await res.json();
  const text: string = body?.choices?.[0]?.message?.content ?? '';
  if (opts.onDelta && text) opts.onDelta(text);

  return {
    text,
    provider: body?.teamz?.provider ?? 'unknown',
    model: body?.model ?? agent.model,
    routing: body?.teamz?.routing ?? { provider: 'unknown', model: '', reason: '', score: 0, candidates: [] },
    hops: body?.teamz?.hops ?? [],
    latencyMs: body?.teamz?.latencyMs ?? Math.round(performance.now() - started),
    usage: {
      promptTokens: body?.usage?.prompt_tokens ?? 0,
      completionTokens: body?.usage?.completion_tokens ?? 0,
      totalTokens: body?.usage?.total_tokens ?? 0,
    },
  };
}
