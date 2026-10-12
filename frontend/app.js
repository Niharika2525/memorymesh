/**
 * MemoryMesh — Voice-First Autonomous Multi-Agent System
 * Frontend Application Controller
 */

// ==========================================================================
// 1. Configuration & Utilities
// ==========================================================================

const DEFAULT_API_BASE = (window.location.protocol.startsWith('http') && window.location.port === '8000')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

let apiBaseUrl = localStorage.getItem('memorymesh_api_base_url') || DEFAULT_API_BASE;

// Application State
const appState = {
  tasks: [],
  verifiedCount: 0,
  speechRecognition: null,
  isRecording: false,
  healthCache: {
    backend: null,
    qdrant: null,
    agents: null,
    omi: null,
  },
};

// Safe HTML Escaper to prevent XSS
function escapeHTML(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Toast Notifications
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px) scale(0.95)';
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

// ==========================================================================
// 2. Client-Side SHA-256 Word-Hash Embedder (Matches backend LLMService)
// ==========================================================================

async function generateDeterministicEmbedding(text, dimensions = 1536) {
  if (!text || !text.trim()) {
    return new Array(dimensions).fill(0.0);
  }

  const words = text.toLowerCase().trim().split(/\s+/);
  const vec = new Array(dimensions).fill(0.0);

  for (const word of words) {
    const encoder = new TextEncoder();
    const data = encoder.encode(word);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    
    // Take high 64 bits as a BigInt for modulo
    let hex = '';
    for (let i = 0; i < 8; i++) {
      hex += hashArray[i].toString(16).padStart(2, '0');
    }
    const hashBigInt = BigInt('0x' + hex);
    const dimBigInt = BigInt(dimensions);

    const idx = Number(hashBigInt % dimBigInt);
    const quotient = Number((hashBigInt / dimBigInt) % 2n);
    const sign = (quotient === 0) ? 1.0 : -1.0;
    vec[idx] += sign;
  }

  // L2 Normalization
  let sumSq = 0.0;
  for (let i = 0; i < dimensions; i++) {
    sumSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(sumSq) || 1.0;

  return vec.map(x => Number((x / norm).toFixed(6)));
}

// ==========================================================================
// 3. API Client Wrapper
// ==========================================================================

async function apiFetch(endpoint, options = {}) {
  const url = `${apiBaseUrl.replace(/\/$/, '')}${endpoint}`;
  const timeoutMs = options.timeout || 45000;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const defaultHeaders = {
    'Accept': 'application/json',
  };

  if (options.body && typeof options.body === 'string') {
    defaultHeaders['Content-Type'] = 'application/json';
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers: { ...defaultHeaders, ...options.headers },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    const contentType = response.headers.get('content-type') || '';
    let data = null;
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      const errorMsg = (data && data.detail)
        ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
        : `Server returned HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMsg);
    }

    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
    }
    throw err;
  }
}

// ==========================================================================
// 4. Navigation & Views Controller
// ==========================================================================

const VIEW_TITLES = {
  overviewView: { title: 'Overview', subtitle: 'Real-time Autonomous Multi-Agent Operations' },
  agentView: { title: 'Agent Workspace', subtitle: 'Collaborative Multi-Agent Task Orchestration' },
  memoryView: { title: 'Persistent Memory Explorer', subtitle: 'Semantic Vector Retrieval & Storage in Qdrant' },
  voiceView: { title: 'Voice Interface', subtitle: 'Real-Time Audio Capture & Omi Webhook Processing' },
  healthView: { title: 'System Diagnostics', subtitle: 'Health Status of FastAPI, Qdrant, Agents, and Omi' },
};

function navigateTo(viewId) {
  const views = document.querySelectorAll('.view-panel');
  views.forEach(v => v.classList.remove('active'));

  const target = document.getElementById(viewId);
  if (target) {
    target.classList.add('active');
  }

  // Update navigation button active states
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    if (item.dataset.target === viewId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Update View Header
  const titleInfo = VIEW_TITLES[viewId] || { title: 'MemoryMesh', subtitle: '' };
  document.getElementById('currentViewTitle').textContent = titleInfo.title;
  document.getElementById('currentViewSubtitle').textContent = titleInfo.subtitle;

  // Auto-close mobile drawer if open
  document.getElementById('sidebar').classList.remove('open');
}
window.navigateTo = navigateTo;

// ==========================================================================
// 5. System Health Controller
// ==========================================================================

async function checkSystemHealth() {
  const timestamp = new Date().toLocaleTimeString();
  const lastCheckedEl = document.getElementById('healthLastChecked');
  if (lastCheckedEl) lastCheckedEl.textContent = `Last checked: ${timestamp}`;

  // 1. Backend Health: GET /health
  try {
    const backendData = await apiFetch('/health', { timeout: 8000 });
    appState.healthCache.backend = backendData;
    updatePill('pillBackend', 'Connected', 'pill-connected');
    updateHealthCard('cardHealthBackend', 'Healthy', 'dot-healthy', 'badge-healthy');
    const mobileDot = document.getElementById('mobileHealthDot');
    if (mobileDot) {
      mobileDot.className = 'status-dot dot-healthy';
      mobileDot.title = 'Backend Connected';
    }
    setElementText('healthBackendApp', backendData.app || 'MemoryMesh');
    setElementText('healthBackendVersion', backendData.version || '0.1.0');
    setElementText('healthBackendEnv', backendData.environment || 'development');
    setElementText('sidebarBackendStatus', 'Connected', 'badge-connected');
  } catch (err) {
    appState.healthCache.backend = null;
    updatePill('pillBackend', 'Disconnected', 'pill-disconnected');
    updateHealthCard('cardHealthBackend', 'Offline', 'dot-disconnected', 'badge-disconnected');
    const mobileDot = document.getElementById('mobileHealthDot');
    if (mobileDot) {
      mobileDot.className = 'status-dot dot-disconnected';
      mobileDot.title = 'Backend Offline';
    }
    setElementText('healthBackendApp', 'Unavailable');
    setElementText('healthBackendVersion', '-');
    setElementText('healthBackendEnv', '-');
    setElementText('sidebarBackendStatus', 'Offline', 'badge-disconnected');
  }

  // 2. Qdrant Health: GET /health/qdrant
  try {
    const qdrantData = await apiFetch('/health/qdrant', { timeout: 8000 });
    appState.healthCache.qdrant = qdrantData;
    const isConn = qdrantData.connected === true;
    updatePill('pillQdrant', isConn ? 'Connected' : 'Disconnected', isConn ? 'pill-connected' : 'pill-disconnected');
    updateHealthCard('cardHealthQdrant', isConn ? 'Connected' : 'Error', isConn ? 'dot-connected' : 'dot-disconnected', isConn ? 'badge-connected' : 'badge-disconnected');
    
    setElementText('healthQdrantUrl', qdrantData.url || 'Configured Endpoint');
    setElementText('healthQdrantCollection', qdrantData.collection || '-');
    setElementText('healthQdrantCollectionExists', qdrantData.collection_exists ? 'Yes' : 'No');
    setElementText('healthQdrantTotalCount', qdrantData.collections_count !== undefined ? String(qdrantData.collections_count) : '-');

    // Update Overview Metric
    setElementText('metricMemoryStatus', isConn ? 'Online' : 'Offline');
    setElementText('metricMemoryCollection', `Collection: ${qdrantData.collection || 'memorymesh_memories'}`);
  } catch (err) {
    appState.healthCache.qdrant = null;
    updatePill('pillQdrant', 'Disconnected', 'pill-disconnected');
    updateHealthCard('cardHealthQdrant', 'Disconnected', 'dot-disconnected', 'badge-disconnected');
    setElementText('metricMemoryStatus', 'Offline');
    setElementText('healthQdrantUrl', 'Unreachable');
  }

  // 3. Agent Mesh Status: GET /api/agent/status
  try {
    const agentData = await apiFetch('/api/agent/status', { timeout: 8000 });
    appState.healthCache.agents = agentData;
    const isReady = agentData.status === 'ready';
    updateHealthCard('cardHealthAgents', isReady ? 'Ready' : 'Unknown', isReady ? 'dot-ready' : 'dot-unknown', isReady ? 'badge-ready' : 'badge-unknown');
    setElementText('healthAgentStatus', agentData.status || 'Ready');
    setElementText('healthAgentList', Array.isArray(agentData.agents) ? agentData.agents.join(', ') : '6 agents');
    setElementText('healthAgentMemoryConnected', agentData.memory_connected ? 'True' : 'False');
  } catch (err) {
    appState.healthCache.agents = null;
    updateHealthCard('cardHealthAgents', 'Unavailable', 'dot-disconnected', 'badge-disconnected');
    setElementText('healthAgentStatus', 'Error');
    setElementText('healthAgentList', '-');
    setElementText('healthAgentMemoryConnected', 'False');
  }

  // 4. Omi Voice Listener: GET /api/omi/status
  try {
    const omiData = await apiFetch('/api/omi/status', { timeout: 8000 });
    appState.healthCache.omi = omiData;
    const isActive = omiData.status === 'active';
    updatePill('pillOmi', isActive ? 'Active' : 'Inactive', isActive ? 'pill-connected' : 'pill-disconnected');
    updateHealthCard('cardHealthOmi', isActive ? 'Active' : 'Inactive', isActive ? 'dot-healthy' : 'dot-unknown', isActive ? 'badge-healthy' : 'badge-unknown');
    
    setElementText('healthOmiStatus', omiData.status || 'Active');
    setElementText('healthOmiService', omiData.service || 'OmiVoiceListener');
    setElementText('healthOmiAppId', omiData.has_app_id ? 'Configured' : 'Not Configured');

    // Overview & Voice Panels
    setElementText('omiServiceName', omiData.service || 'OmiVoiceListener');
    setElementText('omiHasAppId', omiData.has_app_id ? 'Configured' : 'Not Set');
    setElementText('omiServiceStatusBadge', isActive ? 'Active' : 'Inactive', isActive ? 'badge-healthy' : 'badge-unknown');
  } catch (err) {
    appState.healthCache.omi = null;
    updatePill('pillOmi', 'Disconnected', 'pill-disconnected');
    updateHealthCard('cardHealthOmi', 'Offline', 'dot-disconnected', 'badge-disconnected');
    setElementText('healthOmiStatus', 'Unreachable');
    setElementText('omiServiceStatusBadge', 'Offline', 'badge-error');
  }

  // 5. Lyzr Agent Studio: GET /health/lyzr
  try {
    const lyzrData = await apiFetch('/health/lyzr', { timeout: 8000 });
    const isLyzrConfigured = lyzrData.configured === true;
    updateHealthCard(
      'cardHealthLyzr',
      isLyzrConfigured ? 'Connected' : 'Local Fallback',
      isLyzrConfigured ? 'dot-healthy' : 'dot-unknown',
      isLyzrConfigured ? 'badge-healthy' : 'badge-outline'
    );
    setElementText('healthLyzrConfigured', isLyzrConfigured ? 'Cloud Active' : 'Unconfigured (Local)');
    setElementText('healthLyzrAgentId', lyzrData.agent_id_preview || 'Not set in .env');
  } catch (err) {
    updateHealthCard('cardHealthLyzr', 'Offline', 'dot-disconnected', 'badge-disconnected');
    setElementText('healthLyzrConfigured', 'Offline');
  }
}

function updatePill(pillId, valText, className) {
  const pill = document.getElementById(pillId);
  if (!pill) return;
  pill.className = `status-pill ${className}`;
  const valEl = pill.querySelector('.pill-val');
  if (valEl) valEl.textContent = valText;
}

function updateHealthCard(cardId, statusText, dotClass, badgeClass) {
  const card = document.getElementById(cardId);
  if (!card) return;
  const dot = card.querySelector('.health-indicator-dot');
  if (dot) dot.className = `health-indicator-dot ${dotClass}`;
  const badge = card.querySelector('.health-status-text');
  if (badge) {
    badge.className = `badge ${badgeClass} health-status-text`;
    badge.textContent = statusText;
  }
}

function setElementText(id, text, badgeClass = null) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (badgeClass) {
    el.className = `badge ${badgeClass}`;
  }
}

// ==========================================================================
// 6. Agent Workspace & Execution Controller
// ==========================================================================

async function executeAgentWorkflow(taskPrompt, speaker = 'user', source = 'dashboard', autoPersist = true) {
  const container = document.getElementById('agentResultsContainer');
  const badgesContainer = document.getElementById('inspectorBadges');
  const metaSubtitle = document.getElementById('inspectorExecutionMeta');

  if (metaSubtitle) metaSubtitle.textContent = `Executing: "${taskPrompt.slice(0, 30)}..."`;
  if (badgesContainer) badgesContainer.innerHTML = '<span class="badge badge-accent">Running Pipeline</span>';

  // Highlight Workflow Nodes
  setWorkflowHighlight(true);

  // Render Loading State in Container
  if (container) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="btn-spinner" style="width: 36px; height: 36px; border-width: 3px; margin-bottom: 1.25rem;"></div>
        <p class="empty-title">Orchestrating Multi-Agent Mesh</p>
        <p class="empty-subtitle text-muted">Lyzr Automata is dispatching tasks to Planner, Memory, Research, Action, and Verifier agents...</p>
      </div>
    `;
  }

  try {
    const payload = {
      task: taskPrompt,
      speaker: speaker || 'user',
      source: source || 'dashboard',
      auto_persist_memory: Boolean(autoPersist),
    };

    const result = await apiFetch('/api/agent/run', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 90000,
    });

    // Record session task
    recordSessionTask(result);

    // Render Detailed Result
    renderAgentExecutionResult(result);
    showToast('Task executed successfully!', 'success');
    return result;
  } catch (err) {
    if (badgesContainer) badgesContainer.innerHTML = '<span class="badge badge-error">Failed</span>';
    if (container) {
      container.innerHTML = `
        <div class="card" style="border-color: rgba(244, 63, 94, 0.4); background: rgba(244, 63, 94, 0.05);">
          <h4 style="color: var(--accent-rose); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            Workflow Execution Failed
          </h4>
          <p class="text-sm" style="color: var(--text-primary); margin-bottom: 0.75rem;">${escapeHTML(err.message)}</p>
          <div class="text-xs text-muted">Verify your FastAPI server is online and backend dependencies are installed.</div>
        </div>
      `;
    }
    showToast(`Execution failed: ${err.message}`, 'error', 5000);
    throw err;
  } finally {
    setWorkflowHighlight(false);
  }
}

function setWorkflowHighlight(active) {
  const nodes = document.querySelectorAll('.wf-node, .wf-subnode');
  nodes.forEach(node => {
    if (active) {
      node.style.borderColor = 'var(--accent-blue-light)';
    } else {
      node.style.borderColor = '';
    }
  });
}

function renderAgentExecutionResult(data) {
  const container = document.getElementById('agentResultsContainer');
  const badgesContainer = document.getElementById('inspectorBadges');
  const metaSubtitle = document.getElementById('inspectorExecutionMeta');

  if (!container) return;

  const isVerified = data.verification && data.verification.verified;
  const executionId = data.execution_id || 'unknown';

  if (metaSubtitle) {
    metaSubtitle.textContent = `ID: ${executionId.slice(0, 8)}... | Speaker: ${escapeHTML(data.speaker || 'user')}`;
  }

  const providerLabel = data.llm_provider === 'lyzr'
    ? '<span class="badge badge-accent">⚡ Lyzr Cloud AI</span>'
    : (data.llm_provider === 'openai'
      ? '<span class="badge badge-accent">🤖 OpenAI</span>'
      : '<span class="badge badge-outline">⚙ Local Fallback</span>');

  if (badgesContainer) {
    badgesContainer.innerHTML = `
      <span class="badge ${isVerified ? 'badge-healthy' : 'badge-error'}">
        ${isVerified ? '✓ Verified' : '⚠ Unverified'}
      </span>
      ${providerLabel}
      <span class="badge badge-outline">${escapeHTML(data.source || 'api')}</span>
    `;
  }

  // Build Output Sections
  let planHtml = '';
  if (data.plan && Array.isArray(data.plan.steps)) {
    const stepsList = data.plan.steps.map(s => `
      <div class="plan-step-item">
        <span class="step-num">${s.step || '#'}</span>
        <div>
          <strong style="color: var(--accent-violet);">${escapeHTML(s.agent || '')}:</strong>
          <span>${escapeHTML(s.description || s.action || '')}</span>
        </div>
      </div>
    `).join('');
    planHtml = `
      <div class="subagent-box">
        <div class="subagent-box-title">
          <span>1. Planner Agent</span>
          <span class="badge badge-outline">${data.plan.steps.length} Steps</span>
        </div>
        <p class="text-xs text-muted mb-2">${escapeHTML(data.plan.plan_summary || 'Sequential plan generated.')}</p>
        <div class="plan-steps-list">${stepsList}</div>
      </div>
    `;
  } else {
    planHtml = `
      <div class="subagent-box">
        <div class="subagent-box-title"><span>1. Planner Agent</span></div>
        <p class="text-xs text-muted">No structured steps provided.</p>
      </div>
    `;
  }

  // Research Result Box
  const researchFindings = data.research ? (data.research.findings || 'Completed') : 'No research executed.';
  const researchHtml = `
    <div class="subagent-box">
      <div class="subagent-box-title">
        <span>2. Research & Reasoning Agent</span>
        <span class="badge ${data.research && data.research.status === 'completed' ? 'badge-healthy' : 'badge-outline'}">
          ${escapeHTML(data.research ? data.research.status : 'N/A')}
        </span>
      </div>
      <p class="text-xs" style="color: var(--text-primary); white-space: pre-wrap;">${escapeHTML(researchFindings)}</p>
    </div>
  `;

  // Action Result Box
  let actionContent = '';
  if (data.action) {
    if (data.action.result !== undefined) {
      actionContent = typeof data.action.result === 'object'
        ? JSON.stringify(data.action.result, null, 2)
        : String(data.action.result);
    } else if (data.action.error) {
      actionContent = `Error: ${data.action.error}`;
    }
  }
  const actionHtml = `
    <div class="subagent-box">
      <div class="subagent-box-title">
        <span>3. Action Execution Agent</span>
        <span class="badge ${data.action && data.action.status === 'success' ? 'badge-healthy' : 'badge-error'}">
          ${escapeHTML(data.action ? (data.action.action || 'generic') : 'N/A')}
        </span>
      </div>
      <pre class="code-font text-xs" style="color: var(--accent-blue-light); max-height: 120px; overflow-y: auto;">${escapeHTML(actionContent || 'Action completed.')}</pre>
    </div>
  `;

  // Verifier Result Box
  const verifierNotes = data.verification ? (data.verification.verification_notes || 'Verified') : 'No verification details.';
  const verifierConfidence = data.verification ? Math.round((data.verification.confidence_score || 0) * 100) : 0;
  const verifierHtml = `
    <div class="subagent-box">
      <div class="subagent-box-title">
        <span>4. Verifier Agent</span>
        <span class="badge ${isVerified ? 'badge-healthy' : 'badge-error'}">${verifierConfidence}% Confidence</span>
      </div>
      <p class="text-xs" style="color: var(--text-primary);">${escapeHTML(verifierNotes)}</p>
    </div>
  `;

  // Recalled and Persisted Memory Sections
  const recalledCount = data.recalled_memories_count || 0;
  let memoryMetaHtml = `
    <div class="subagent-box" style="grid-column: 1 / -1;">
      <div class="subagent-box-title">
        <span>Persistent Vector Memory (Qdrant)</span>
        <span class="badge badge-accent">${recalledCount} Recalled</span>
      </div>
      <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.75rem;">
        <div><strong>Recalled Points:</strong> ${recalledCount}</div>
        <div><strong>Persisted Outcome:</strong> ${data.persisted_memory ? `<span style="color: var(--accent-emerald);">Stored (ID: ${escapeHTML(data.persisted_memory.id || '')})</span>` : '<span style="color: var(--text-muted);">Disabled</span>'}</div>
      </div>
    </div>
  `;

  // Combine into Final HTML
  container.innerHTML = `
    <div class="final-response-box">
      <div class="final-response-title">
        <span>Synthesized Multi-Agent Outcome</span>
        <button type="button" class="btn btn-xs btn-ghost" onclick="navigator.clipboard.writeText('${escapeHTML(data.final_response || '')}').then(() => showToast('Copied to clipboard', 'info'))">Copy</button>
      </div>
      <div class="final-response-text">${escapeHTML(data.final_response || 'No response produced.')}</div>
    </div>

    <div class="subagent-grid">
      ${planHtml}
      ${researchHtml}
      ${actionHtml}
      ${verifierHtml}
      ${memoryMetaHtml}
    </div>
  `;
}

function recordSessionTask(result) {
  appState.tasks.unshift({
    id: result.execution_id || String(Date.now()),
    task: result.task || 'Unnamed Task',
    response: result.final_response || '',
    speaker: result.speaker || 'user',
    verified: Boolean(result.verification && result.verification.verified),
    timestamp: new Date().toLocaleTimeString(),
    fullResult: result,
  });

  if (result.verification && result.verification.verified) {
    appState.verifiedCount++;
  }

  updateSessionMetrics();
  renderRecentActivityList();
}

function updateSessionMetrics() {
  const total = appState.tasks.length;
  setElementText('sessionTasksCount', String(total));
  setElementText('metricSessionTasks', String(total));

  const verified = appState.tasks.filter(t => t.verified).length;
  const rate = total > 0 ? Math.round((verified / total) * 100) : 100;

  setElementText('metricVerifiedRate', `${rate}%`);
  setElementText('metricVerifiedCount', `${verified} of ${total} verified`);
}

function renderRecentActivityList() {
  const container = document.getElementById('recentActivityContainer');
  if (!container) return;

  if (appState.tasks.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
        <p class="empty-title">No tasks executed yet</p>
        <p class="empty-subtitle text-muted">Enter a task above or speak a voice command to see live multi-agent execution results.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = appState.tasks.slice(0, 10).map((t, index) => `
    <div class="activity-item">
      <div class="activity-left">
        <span class="badge ${t.verified ? 'badge-healthy' : 'badge-error'}">${t.verified ? '✓' : '⚠'}</span>
        <div style="min-width: 0;">
          <div class="activity-text" title="${escapeHTML(t.task)}">${escapeHTML(t.task)}</div>
          <div class="activity-meta">By ${escapeHTML(t.speaker)} at ${t.timestamp}</div>
        </div>
      </div>
      <button class="btn btn-xs btn-secondary" onclick="viewTaskDetailsModal(${index})">Inspect</button>
    </div>
  `).join('');
}

function viewTaskDetailsModal(index) {
  const taskItem = appState.tasks[index];
  if (!taskItem) return;

  const modal = document.getElementById('taskModal');
  const modalTitle = document.getElementById('modalTaskTitle');
  const modalBody = document.getElementById('modalTaskBody');

  if (modalTitle) modalTitle.textContent = `Task: ${taskItem.task.slice(0, 40)}...`;
  if (modalBody) {
    modalBody.innerHTML = `
      <div style="margin-bottom: 1rem;">
        <span class="badge ${taskItem.verified ? 'badge-healthy' : 'badge-error'}">${taskItem.verified ? 'Verified' : 'Unverified'}</span>
        <span class="text-xs text-muted" style="margin-left: 0.5rem;">Executed at ${taskItem.timestamp}</span>
      </div>
      <div style="margin-bottom: 1rem;">
        <h4 class="text-xs text-muted uppercase">Final Response</h4>
        <p class="text-sm" style="color: var(--text-primary); margin-top: 0.35rem; line-height: 1.5;">${escapeHTML(taskItem.response)}</p>
      </div>
      <div>
        <h4 class="text-xs text-muted uppercase">Raw JSON Inspection</h4>
        <pre class="code-font text-xs" style="background: var(--bg-input); padding: 1rem; border-radius: 8px; max-height: 280px; overflow-y: auto;">${escapeHTML(JSON.stringify(taskItem.fullResult, null, 2))}</pre>
      </div>
    `;
  }

  if (modal) modal.classList.remove('hidden');
}
window.viewTaskDetailsModal = viewTaskDetailsModal;

// ==========================================================================
// 7. Persistent Memory Explorer Controller
// ==========================================================================

async function searchMemories(vectorArray, limit = 5, scoreThreshold = null) {
  const container = document.getElementById('memoryCardsList');
  const countBadge = document.getElementById('memoryCountBadge');
  const contextWrapper = document.getElementById('memoryContextTextWrapper');
  const contextPre = document.getElementById('memoryContextText');

  if (container) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="btn-spinner" style="width: 32px; height: 32px; margin-bottom: 1rem;"></div>
        <p class="empty-title">Querying Qdrant Vector Collection...</p>
      </div>
    `;
  }

  try {
    const payload = {
      vector: vectorArray,
      limit: parseInt(limit, 10) || 5,
    };
    if (scoreThreshold !== null && scoreThreshold !== '') {
      payload.score_threshold = parseFloat(scoreThreshold);
    }

    const data = await apiFetch('/api/memory/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    if (countBadge) countBadge.textContent = `${data.count || 0} Results`;

    // Render Context Text Preview
    if (data.context_text && contextWrapper && contextPre) {
      contextPre.textContent = data.context_text;
      contextWrapper.classList.remove('hidden');
    } else if (contextWrapper) {
      contextWrapper.classList.add('hidden');
    }

    // Render Memory Cards
    if (!data.memories || data.memories.length === 0) {
      if (container) {
        container.innerHTML = `
          <div class="empty-state">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <p class="empty-title">No Matching Memories Found</p>
            <p class="empty-subtitle text-muted">Try lowering the score threshold or auto-generating a new embedding vector.</p>
          </div>
        `;
      }
      return;
    }

    if (container) {
      container.innerHTML = data.memories.map(m => `
        <div class="memory-point-card" id="memCard-${escapeHTML(m.id)}">
          <div class="memory-card-header">
            <div>
              <span class="badge badge-accent">${m.score !== undefined ? `Similarity: ${(m.score * 100).toFixed(1)}%` : 'Point'}</span>
              <span class="code-font text-xs text-muted" style="margin-left: 0.5rem;">ID: ${escapeHTML(m.id)}</span>
            </div>
            <button class="btn btn-xs btn-ghost" style="color: var(--accent-rose);" onclick="deleteMemoryPoint('${escapeHTML(m.id)}')">Delete</button>
          </div>
          <div class="memory-content-text">${escapeHTML(m.content || '')}</div>
          <div class="memory-meta-tags">
            ${m.created_at ? `<span class="meta-tag">🕒 ${escapeHTML(m.created_at)}</span>` : ''}
            ${m.metadata ? Object.entries(m.metadata).map(([k, v]) => `<span class="meta-tag">${escapeHTML(k)}: ${escapeHTML(String(v))}</span>`).join('') : ''}
          </div>
        </div>
      `).join('');
    }

    showToast(`Found ${data.count || 0} semantic memories`, 'success');
  } catch (err) {
    if (container) {
      container.innerHTML = `
        <div class="card" style="border-color: rgba(244, 63, 94, 0.4); background: rgba(244, 63, 94, 0.05);">
          <h4 style="color: var(--accent-rose); margin-bottom: 0.5rem;">Search Failed</h4>
          <p class="text-sm">${escapeHTML(err.message)}</p>
        </div>
      `;
    }
    showToast(`Search failed: ${err.message}`, 'error');
  }
}

async function storeMemoryPoint(content, vectorArray, customId = null, metadata = {}) {
  try {
    const payload = {
      content: content,
      vector: vectorArray,
      metadata: metadata || {},
    };
    if (customId && customId.trim()) {
      payload.id = customId.trim();
    }

    const res = await apiFetch('/api/memory/store', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    showToast(`Memory point saved! ID: ${res.id || 'stored'}`, 'success');
    return res;
  } catch (err) {
    showToast(`Failed to store memory: ${err.message}`, 'error', 4500);
    throw err;
  }
}

async function fetchMemoryById(memoryId) {
  if (!memoryId || !memoryId.trim()) {
    showToast('Please enter a memory point ID', 'error');
    return;
  }
  const container = document.getElementById('memoryCardsList');
  try {
    const item = await apiFetch(`/api/memory/${encodeURIComponent(memoryId.trim())}`);
    if (container) {
      container.innerHTML = `
        <div class="memory-point-card">
          <div class="memory-card-header">
            <div>
              <span class="badge badge-accent">Direct Match</span>
              <span class="code-font text-xs text-muted" style="margin-left: 0.5rem;">ID: ${escapeHTML(item.id)}</span>
            </div>
            <button class="btn btn-xs btn-ghost" style="color: var(--accent-rose);" onclick="deleteMemoryPoint('${escapeHTML(item.id)}')">Delete</button>
          </div>
          <div class="memory-content-text">${escapeHTML(item.content || '')}</div>
          <div class="memory-meta-tags">
            ${item.created_at ? `<span class="meta-tag">🕒 ${escapeHTML(item.created_at)}</span>` : ''}
            ${item.metadata ? Object.entries(item.metadata).map(([k, v]) => `<span class="meta-tag">${escapeHTML(k)}: ${escapeHTML(String(v))}</span>`).join('') : ''}
          </div>
        </div>
      `;
    }
    showToast('Memory retrieved', 'success');
  } catch (err) {
    showToast(`Fetch failed: ${err.message}`, 'error');
  }
}
window.fetchMemoryById = fetchMemoryById;

async function deleteMemoryPoint(memoryId) {
  if (!confirm(`Are you sure you want to delete memory point "${memoryId}"?`)) return;
  try {
    await apiFetch(`/api/memory/${encodeURIComponent(memoryId)}`, { method: 'DELETE' });
    const card = document.getElementById(`memCard-${memoryId}`);
    if (card) card.remove();
    showToast('Memory point deleted from Qdrant', 'success');
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, 'error');
  }
}
window.deleteMemoryPoint = deleteMemoryPoint;

// ==========================================================================
// 8. Voice Interface Controller
// ==========================================================================

function initVoiceRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const badge = document.getElementById('speechSupportBadge');
  const micBtn = document.getElementById('micCaptureBtn');
  const statusLabel = document.getElementById('voiceStatusLabel');

  if (!SpeechRecognition) {
    if (badge) {
      badge.textContent = 'Web Speech Unsupported';
      badge.className = 'badge badge-error';
    }
    if (statusLabel) {
      statusLabel.textContent = 'Speech recognition not supported in this browser. You can type or paste transcripts.';
    }
    if (micBtn) micBtn.disabled = true;
    return;
  }

  if (badge) {
    badge.textContent = 'Web Speech Ready';
    badge.className = 'badge badge-healthy';
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    appState.isRecording = true;
    if (micBtn) micBtn.classList.add('listening');
    if (statusLabel) statusLabel.textContent = 'Listening to your voice... Speak now.';
  };

  recognition.onresult = (event) => {
    let transcriptText = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcriptText += event.results[i][0].transcript;
    }
    const transcriptInput = document.getElementById('voiceTranscriptText');
    if (transcriptInput) {
      transcriptInput.value = transcriptText;
      updateVoiceSubmitButton();
    }
  };

  recognition.onerror = (event) => {
    if (event.error !== 'no-speech') {
      showToast(`Speech recognition error: ${event.error}`, 'error');
    }
    stopListening();
  };

  recognition.onend = () => {
    stopListening();
  };

  appState.speechRecognition = recognition;

  if (micBtn) {
    micBtn.addEventListener('click', () => {
      if (appState.isRecording) {
        stopListening();
      } else {
        startListening();
      }
    });
  }
}

function startListening() {
  if (appState.speechRecognition && !appState.isRecording) {
    try {
      appState.speechRecognition.start();
    } catch (err) {
      console.warn('Recognition start exception:', err);
    }
  }
}

function stopListening() {
  appState.isRecording = false;
  const micBtn = document.getElementById('micCaptureBtn');
  const statusLabel = document.getElementById('voiceStatusLabel');
  if (micBtn) micBtn.classList.remove('listening');
  if (statusLabel) statusLabel.textContent = 'Click microphone to speak';
  if (appState.speechRecognition) {
    try {
      appState.speechRecognition.stop();
    } catch (e) {}
  }
}

function updateVoiceSubmitButton() {
  const input = document.getElementById('voiceTranscriptText');
  const btn = document.getElementById('submitVoiceTranscriptBtn');
  if (btn && input) {
    btn.disabled = !Boolean(input.value && input.value.trim());
  }
}

async function submitVoiceTranscript() {
  const transcriptInput = document.getElementById('voiceTranscriptText');
  const speakerInput = document.getElementById('voiceSpeakerInput');
  const sessionInput = document.getElementById('voiceSessionId');
  const outputBox = document.getElementById('speechOutputBox');
  const outputContent = document.getElementById('speechOutputContent');

  if (!transcriptInput || !transcriptInput.value.trim()) return;

  const payload = {
    transcript: transcriptInput.value.trim(),
    speaker: (speakerInput && speakerInput.value.trim()) || 'user',
    session_id: (sessionInput && sessionInput.value.trim()) || null,
  };

  try {
    showToast('Transmitting voice command to mesh...', 'info');
    const result = await apiFetch('/api/omi/voice', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    if (result.status === 'success') {
      showToast('Voice instruction processed!', 'success');
      const speechRes = result.speech_response || (result.orchestrator_result && result.orchestrator_result.final_response) || '';
      
      if (outputBox && outputContent) {
        outputContent.textContent = speechRes;
        outputBox.classList.remove('hidden');
      }

      // Record session task if orchestrator result exists
      if (result.orchestrator_result) {
        recordSessionTask(result.orchestrator_result);
      }
    } else {
      showToast(result.message || 'Instruction was ignored', 'info');
    }
  } catch (err) {
    showToast(`Voice submission failed: ${err.message}`, 'error');
  }
}

function speakAgentResponse() {
  const outputContent = document.getElementById('speechOutputContent');
  if (!outputContent || !outputContent.textContent) return;

  if (!('speechSynthesis' in window)) {
    showToast('Text-to-speech not supported in this browser', 'error');
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(outputContent.textContent);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// ==========================================================================
// 9. Event Listeners & Initialization
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Navigation item clicks
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const target = item.dataset.target;
      if (target) navigateTo(target);
    });
  });

  // Mobile Menu Toggle
  const mobileBtn = document.getElementById('mobileMenuBtn');
  const sidebar = document.getElementById('sidebar');
  if (mobileBtn && sidebar) {
    mobileBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }

  // Quick Refresh Button
  const refreshBtn = document.getElementById('quickRefreshBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', () => checkSystemHealth());
  const healthRefreshBtn = document.getElementById('healthRefreshAllBtn');
  if (healthRefreshBtn) healthRefreshBtn.addEventListener('click', () => checkSystemHealth());

  // Preset Chips on Overview
  document.querySelectorAll('.chip-preset[data-text]').forEach(chip => {
    chip.addEventListener('click', () => {
      const input = document.getElementById('overviewTaskInput');
      if (input) input.value = chip.dataset.text;
    });
  });

  // Overview Quick Task Form
  const overviewForm = document.getElementById('overviewTaskForm');
  if (overviewForm) {
    overviewForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = document.getElementById('overviewTaskInput');
      if (!input || !input.value.trim()) return;
      const task = input.value.trim();

      // Switch to Agent Workspace and execute
      navigateTo('agentView');
      const agentInput = document.getElementById('agentTaskPrompt');
      if (agentInput) agentInput.value = task;
      await executeAgentWorkflow(task, 'user', 'dashboard', true);
    });
  }

  // Clear Activity Log Button
  const clearActivityBtn = document.getElementById('clearActivityBtn');
  if (clearActivityBtn) {
    clearActivityBtn.addEventListener('click', () => {
      appState.tasks = [];
      appState.verifiedCount = 0;
      updateSessionMetrics();
      renderRecentActivityList();
      showToast('Session log cleared', 'info');
    });
  }

  // Agent Task Form
  const agentForm = document.getElementById('agentTaskForm');
  if (agentForm) {
    agentForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const promptInput = document.getElementById('agentTaskPrompt');
      const speakerInput = document.getElementById('agentSpeaker');
      const sourceSelect = document.getElementById('agentSource');
      const persistToggle = document.getElementById('agentAutoPersist');
      const submitBtn = document.getElementById('agentSubmitBtn');

      if (!promptInput || !promptInput.value.trim()) return;

      const prompt = promptInput.value.trim();
      const speaker = (speakerInput && speakerInput.value.trim()) || 'user';
      const source = (sourceSelect && sourceSelect.value) || 'dashboard';
      const autoPersist = persistToggle ? persistToggle.checked : true;

      if (submitBtn) {
        submitBtn.disabled = true;
        const spinner = submitBtn.querySelector('.btn-spinner');
        if (spinner) spinner.classList.remove('hidden');
      }

      try {
        await executeAgentWorkflow(prompt, speaker, source, autoPersist);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          const spinner = submitBtn.querySelector('.btn-spinner');
          if (spinner) spinner.classList.add('hidden');
        }
      }
    });
  }

  // Memory Search Form & Vector Helpers
  const memorySearchForm = document.getElementById('memorySearchForm');
  if (memorySearchForm) {
    memorySearchForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const vectorInput = document.getElementById('searchVectorInput');
      const limitInput = document.getElementById('searchLimit');
      const thresholdInput = document.getElementById('searchScoreThreshold');

      if (!vectorInput || !vectorInput.value.trim()) {
        showToast('Please provide a vector embedding', 'error');
        return;
      }

      let parsedVector = null;
      try {
        const raw = vectorInput.value.trim();
        parsedVector = raw.startsWith('[') ? JSON.parse(raw) : raw.split(',').map(s => parseFloat(s.trim()));
        if (!Array.isArray(parsedVector) || parsedVector.some(isNaN)) {
          throw new Error('Vector contains invalid numbers');
        }
      } catch (err) {
        showToast('Invalid vector format: please provide comma-separated numbers or JSON array', 'error');
        return;
      }

      await searchMemories(parsedVector, limitInput ? limitInput.value : 5, thresholdInput ? thresholdInput.value : null);
    });
  }

  // Memory Auto-Embed Button (Search)
  const genSearchVecBtn = document.getElementById('generateSearchVectorBtn');
  if (genSearchVecBtn) {
    genSearchVecBtn.addEventListener('click', async () => {
      const textInput = document.getElementById('searchQueryTextInput');
      const vectorInput = document.getElementById('searchVectorInput');
      if (!textInput || !textInput.value.trim()) {
        showToast('Please enter text to embed', 'error');
        return;
      }
      showToast('Generating deterministic SHA-256 embedding...', 'info');
      const vec = await generateDeterministicEmbedding(textInput.value.trim(), 1536);
      if (vectorInput) vectorInput.value = JSON.stringify(vec);
      showToast('Embedding generated (1536D)', 'success');
    });
  }

  // Preset Vector Buttons
  const preset4dBtn = document.getElementById('preset4dVectorBtn');
  if (preset4dBtn) {
    preset4dBtn.addEventListener('click', () => {
      const vecInput = document.getElementById('searchVectorInput');
      if (vecInput) vecInput.value = '[1.0, 0.0, 0.0, 0.0]';
    });
  }

  const presetRandomEmbedBtn = document.getElementById('presetRandomEmbedBtn');
  if (presetRandomEmbedBtn) {
    presetRandomEmbedBtn.addEventListener('click', async () => {
      const vec = await generateDeterministicEmbedding('hackathon project deadline and deliverables', 1536);
      const vecInput = document.getElementById('searchVectorInput');
      if (vecInput) vecInput.value = JSON.stringify(vec);
      showToast('Sample embedding populated', 'info');
    });
  }

  // Memory Store Form
  const memoryStoreForm = document.getElementById('memoryStoreForm');
  if (memoryStoreForm) {
    memoryStoreForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const contentInput = document.getElementById('storeContentInput');
      const vectorInput = document.getElementById('storeVectorInput');
      const customIdInput = document.getElementById('storeCustomId');
      const metadataInput = document.getElementById('storeMetadataJson');

      if (!contentInput || !contentInput.value.trim()) {
        showToast('Memory content text is required', 'error');
        return;
      }

      let parsedVector = null;
      try {
        const raw = vectorInput ? vectorInput.value.trim() : '';
        if (raw) {
          parsedVector = raw.startsWith('[') ? JSON.parse(raw) : raw.split(',').map(s => parseFloat(s.trim()));
        } else {
          parsedVector = await generateDeterministicEmbedding(contentInput.value.trim(), 1536);
        }
      } catch (err) {
        showToast('Invalid vector format', 'error');
        return;
      }

      let parsedMeta = {};
      try {
        if (metadataInput && metadataInput.value.trim()) {
          parsedMeta = JSON.parse(metadataInput.value.trim());
        }
      } catch (err) {
        showToast('Metadata must be valid JSON', 'error');
        return;
      }

      await storeMemoryPoint(
        contentInput.value.trim(),
        parsedVector,
        customIdInput ? customIdInput.value : null,
        parsedMeta
      );
    });
  }

  // Memory Auto-Embed Button (Store)
  const genStoreVecBtn = document.getElementById('generateStoreVectorBtn');
  if (genStoreVecBtn) {
    genStoreVecBtn.addEventListener('click', async () => {
      const contentInput = document.getElementById('storeContentInput');
      const vectorInput = document.getElementById('storeVectorInput');
      if (!contentInput || !contentInput.value.trim()) {
        showToast('Please enter memory content first', 'error');
        return;
      }
      const vec = await generateDeterministicEmbedding(contentInput.value.trim(), 1536);
      if (vectorInput) vectorInput.value = JSON.stringify(vec);
      showToast('Embedding generated from content (1536D)', 'success');
    });
  }

  // Fetch Memory By ID Button
  const fetchMemBtn = document.getElementById('fetchMemoryIdBtn');
  if (fetchMemBtn) {
    fetchMemBtn.addEventListener('click', () => {
      const input = document.getElementById('fetchMemoryIdInput');
      if (input) fetchMemoryById(input.value);
    });
  }

  // Copy Context Button
  const copyContextBtn = document.getElementById('copyContextBtn');
  if (copyContextBtn) {
    copyContextBtn.addEventListener('click', () => {
      const pre = document.getElementById('memoryContextText');
      if (pre && pre.textContent) {
        navigator.clipboard.writeText(pre.textContent).then(() => showToast('Context text copied', 'info'));
      }
    });
  }

  // Voice Transcript Input & Button
  const voiceTranscriptText = document.getElementById('voiceTranscriptText');
  if (voiceTranscriptText) {
    voiceTranscriptText.addEventListener('input', updateVoiceSubmitButton);
  }

  const submitVoiceBtn = document.getElementById('submitVoiceTranscriptBtn');
  if (submitVoiceBtn) {
    submitVoiceBtn.addEventListener('click', submitVoiceTranscript);
  }

  const ttsBtn = document.getElementById('ttsSpeakBtn');
  if (ttsBtn) {
    ttsBtn.addEventListener('click', speakAgentResponse);
  }

  // Webhook Simulator Button
  const webhookSimBtn = document.getElementById('sendWebhookSimBtn');
  if (webhookSimBtn) {
    webhookSimBtn.addEventListener('click', async () => {
      const editor = document.getElementById('webhookPayloadEditor');
      if (!editor || !editor.value.trim()) return;

      try {
        const payload = JSON.parse(editor.value.trim());
        showToast('Posting simulated Omi webhook...', 'info');
        const res = await apiFetch('/api/omi/webhook', {
          method: 'POST',
          body: JSON.stringify(payload),
        });

        if (res.status === 'success') {
          showToast('Webhook handled by agent mesh!', 'success');
          if (res.orchestrator_result) {
            recordSessionTask(res.orchestrator_result);
          }
        } else {
          showToast(`Webhook response: ${res.status} (${res.message || ''})`, 'info');
        }
      } catch (err) {
        showToast(`Webhook simulation error: ${err.message}`, 'error');
      }
    });
  }

  // Base URL Configuration
  const customBaseInput = document.getElementById('customApiBaseUrl');
  const saveBaseUrlBtn = document.getElementById('saveApiBaseUrlBtn');
  const resetBaseUrlBtn = document.getElementById('resetApiBaseUrlBtn');
  const sidebarApiUrl = document.getElementById('sidebarApiUrl');

  if (customBaseInput) customBaseInput.value = apiBaseUrl;
  if (sidebarApiUrl) sidebarApiUrl.textContent = apiBaseUrl;

  if (saveBaseUrlBtn && customBaseInput) {
    saveBaseUrlBtn.addEventListener('click', () => {
      const val = customBaseInput.value.trim();
      if (!val) return;
      apiBaseUrl = val;
      localStorage.setItem('memorymesh_api_base_url', apiBaseUrl);
      if (sidebarApiUrl) sidebarApiUrl.textContent = apiBaseUrl;
      showToast(`Base URL saved: ${apiBaseUrl}`, 'success');
      checkSystemHealth();
    });
  }

  if (resetBaseUrlBtn && customBaseInput) {
    resetBaseUrlBtn.addEventListener('click', () => {
      apiBaseUrl = DEFAULT_API_BASE;
      localStorage.removeItem('memorymesh_api_base_url');
      customBaseInput.value = apiBaseUrl;
      if (sidebarApiUrl) sidebarApiUrl.textContent = apiBaseUrl;
      showToast(`Base URL reset to default: ${apiBaseUrl}`, 'info');
      checkSystemHealth();
    });
  }

  // Modal Close Handlers
  const modal = document.getElementById('taskModal');
  const modalClose = document.getElementById('modalCloseBtn');
  if (modalClose && modal) {
    modalClose.addEventListener('click', () => modal.classList.add('hidden'));
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.add('hidden');
    });
  }

  // Initialize Speech Recognition
  initVoiceRecognition();

  // Initial Health Check
  checkSystemHealth();
});

