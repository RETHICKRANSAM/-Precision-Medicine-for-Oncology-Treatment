/**
 * Risk Monitor - SOC Dashboard Application Logic
 * Supports dual-mode: Live Flask REST API + Offline Client-Side ML Simulation (GitHub Pages compatible)
 */

// State
let isBackendOnline = false;
let auditRecords = [];
let activeEvaluation = null;
let currentFilter = 'ALL';

// Chart Instances
let radarChart = null;
let doughnutChart = null;
let timelineChart = null;

// Initial Bootstrap Data for Client-Side Simulation (GitHub Pages)
const MOCK_HISTORICAL_RECORDS = [
  {
    deployment_id: "DEP-10025",
    timestamp: "2026-09-07 14:15:30 UTC",
    version: "v2.5.0",
    commit_sha: "4f89ac2",
    service_name: "payment-service",
    environment: "canary-prod",
    deployer: "sarah.jenkins",
    risk_score: 18,
    decision: "ALLOW",
    color: "emerald",
    status_text: "Deployment Cleared for Production",
    key_factors: ["All telemetry within standard healthy deviation baselines."],
    verified_by: "auto-policy-engine",
    telemetry: { Error_Rate_Delta: 0.04, Latency_P95_ms: 58, Unit_Test_Failure_Rate: 0.0, Security_Critical_CVEs: 0 },
    radar_signals: { "Error Rate": 12, "Latency": 20, "Test Failures": 5, "Vulnerabilities": 0, "Network / Auth": 10, "Code Blast Radius": 15 }
  },
  {
    deployment_id: "DEP-10024",
    timestamp: "2026-09-07 12:45:10 UTC",
    version: "v2.4.9",
    commit_sha: "e29ba71",
    service_name: "search-indexer",
    environment: "canary-prod",
    deployer: "alex.chen",
    risk_score: 64,
    decision: "PAUSE",
    color: "amber",
    status_text: "Manual Engineer Verification Required",
    key_factors: ["High P95 latency: 230ms exceeding 200ms threshold.", "1 High CVE vulnerability detected."],
    verified_by: "pending",
    telemetry: { Error_Rate_Delta: 0.45, Latency_P95_ms: 230, Unit_Test_Failure_Rate: 1.2, Security_Critical_CVEs: 0 },
    radar_signals: { "Error Rate": 35, "Latency": 68, "Test Failures": 25, "Vulnerabilities": 30, "Network / Auth": 40, "Code Blast Radius": 45 }
  },
  {
    deployment_id: "DEP-10023",
    timestamp: "2026-09-07 10:20:00 UTC",
    version: "v2.4.8",
    commit_sha: "9a01f4c",
    service_name: "auth-gateway",
    environment: "canary-prod",
    deployer: "marcus.vance",
    risk_score: 92,
    decision: "BLOCK",
    color: "rose",
    status_text: "Release Halted - Unsafe Telemetry Detected",
    key_factors: ["2 Critical Security CVE vulnerability flag(s) present.", "Elevated canary error rate: +1.85%."],
    verified_by: "auto-policy-engine",
    telemetry: { Error_Rate_Delta: 1.85, Latency_P95_ms: 190, Unit_Test_Failure_Rate: 0.0, Security_Critical_CVEs: 2 },
    radar_signals: { "Error Rate": 55, "Latency": 48, "Test Failures": 10, "Vulnerabilities": 95, "Network / Auth": 75, "Code Blast Radius": 60 }
  },
  {
    deployment_id: "DEP-10022",
    timestamp: "2026-09-07 08:05:22 UTC",
    version: "v2.4.7",
    commit_sha: "d3810ae",
    service_name: "order-processing",
    environment: "canary-prod",
    deployer: "ci-bot-auto",
    risk_score: 98,
    decision: "BLOCK",
    color: "rose",
    status_text: "Release Halted - Unsafe Telemetry Detected",
    key_factors: ["Integration test suite failed during canary execution.", "Unit test failure rate at 14.5%."],
    verified_by: "auto-policy-engine",
    telemetry: { Error_Rate_Delta: 5.4, Latency_P95_ms: 520, Unit_Test_Failure_Rate: 14.5, Security_Critical_CVEs: 1 },
    radar_signals: { "Error Rate": 95, "Latency": 90, "Test Failures": 100, "Vulnerabilities": 65, "Network / Auth": 80, "Code Blast Radius": 85 }
  },
  {
    deployment_id: "DEP-10021",
    timestamp: "2026-09-06 22:11:45 UTC",
    version: "v2.4.6",
    commit_sha: "b492fc1",
    service_name: "user-profile",
    environment: "canary-prod",
    deployer: "elena.rostova",
    risk_score: 22,
    decision: "ALLOW",
    color: "emerald",
    status_text: "Deployment Cleared for Production",
    key_factors: ["All telemetry within standard healthy deviation baselines."],
    verified_by: "auto-policy-engine",
    telemetry: { Error_Rate_Delta: 0.08, Latency_P95_ms: 62, Unit_Test_Failure_Rate: 0.0, Security_Critical_CVEs: 0 },
    radar_signals: { "Error Rate": 15, "Latency": 22, "Test Failures": 5, "Vulnerabilities": 0, "Network / Auth": 15, "Code Blast Radius": 20 }
  }
];

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  initCharts();
  await checkBackendHealth();
  await loadDashboardData();
});

// Check if Flask Backend is available
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health', { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      isBackendOnline = true;
      document.getElementById('connectionBadge').className = 'flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-mono';
      document.getElementById('connectionText').textContent = 'Backend: Flask REST API Online';
      return;
    }
  } catch (e) {
    // Offline / GitHub Pages
  }
  isBackendOnline = false;
  document.getElementById('connectionBadge').className = 'flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-950/40 border border-indigo-500/30 text-indigo-300 text-xs font-mono';
  document.getElementById('connectionText').textContent = 'Client-Side ML Simulation Active';
}

// Load Metrics & Audit Trail
async function loadDashboardData() {
  if (isBackendOnline) {
    try {
      const [sumRes, trailRes] = await Promise.all([
        fetch('/api/metrics/summary'),
        fetch('/api/deployments/audit-trail?limit=50')
      ]);
      const sumData = await sumRes.json();
      const trailData = await trailRes.json();

      auditRecords = trailData.audit_trail || [];
      updateExecutiveStats(sumData);
      
      if (auditRecords.length > 0) {
        renderActiveEvaluation(auditRecords[0]);
      }
      renderAuditTable();
      return;
    } catch (e) {
      console.warn("Falling back to client-side data.", e);
    }
  }

  // Client-Side Fallback Mode
  auditRecords = [...MOCK_HISTORICAL_RECORDS];
  computeClientStats();
  renderActiveEvaluation(auditRecords[0]);
  renderAuditTable();
}

function computeClientStats() {
  const total = auditRecords.length;
  const allows = auditRecords.filter(r => r.decision === 'ALLOW').length;
  const pauses = auditRecords.filter(r => r.decision === 'PAUSE').length;
  const blocks = auditRecords.filter(r => r.decision === 'BLOCK').length;
  const avg = (auditRecords.reduce((acc, r) => acc + r.risk_score, 0) / (total || 1)).toFixed(1);

  updateExecutiveStats({
    total_deployments: total,
    allow_count: allows,
    allow_percentage: Math.round((allows / total) * 100),
    pause_count: pauses,
    pause_percentage: Math.round((pauses / total) * 100),
    block_count: blocks,
    block_percentage: Math.round((blocks / total) * 100),
    average_risk_score: avg
  });
}

function updateExecutiveStats(stats) {
  document.getElementById('statTotal').textContent = stats.total_deployments;
  document.getElementById('statAllow').textContent = stats.allow_count;
  document.getElementById('statAllowPct').textContent = `${stats.allow_percentage}%`;
  document.getElementById('barAllow').style.width = `${stats.allow_percentage}%`;

  document.getElementById('statPause').textContent = stats.pause_count;
  document.getElementById('statPausePct').textContent = `${stats.pause_percentage}%`;
  document.getElementById('barPause').style.width = `${stats.pause_percentage}%`;

  document.getElementById('statBlock').textContent = stats.block_count;
  document.getElementById('statBlockPct').textContent = `${stats.block_percentage}%`;
  document.getElementById('barBlock').style.width = `${stats.block_percentage}%`;

  document.getElementById('statAvgScore').textContent = stats.average_risk_score;
  document.getElementById('barAvgScore').style.width = `${stats.average_risk_score}%`;

  // Update Doughnut Chart
  updateDoughnutChart(stats.allow_count, stats.pause_count, stats.block_count);
}

// Render the Active Canary Evaluation Card
function renderActiveEvaluation(record) {
  activeEvaluation = record;
  const score = record.risk_score;
  const decision = record.decision;

  // Counter animation
  animateScoreCounter(score);

  // SVG Gauge stroke offset: Circumference = 2 * PI * 70 ≈ 440
  const maxOffset = 440;
  const targetOffset = maxOffset - (score / 100) * maxOffset;
  const gaugeProgress = document.getElementById('gaugeProgress');
  gaugeProgress.style.strokeDashoffset = targetOffset;

  const card = document.getElementById('activeEvaluationCard');
  const badge = document.getElementById('decisionBadge');
  const pulseRing = document.getElementById('pulseRing');
  const pulseDot = document.getElementById('pulseDot');
  const btnOverride = document.getElementById('btnOverride');

  // Clear glow classes
  card.classList.remove('glow-allow', 'glow-pause', 'glow-block');

  if (decision === 'ALLOW') {
    gaugeProgress.setAttribute('stroke', '#10b981');
    card.classList.add('glow-allow');
    badge.className = 'py-2.5 px-6 rounded-xl font-bold text-center tracking-wider text-sm transition-all bg-emerald-500/15 text-emerald-400 border border-emerald-500/40 shadow-lg shadow-emerald-500/10';
    badge.textContent = 'ALLOW — PROCEED TO PRODUCTION';
    pulseRing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75';
    pulseDot.className = 'relative inline-flex rounded-full h-3 w-3 bg-emerald-500';
    btnOverride.classList.add('hidden');
    document.getElementById('cardStatusLabel').textContent = 'Canary Cleared & Healthy';
  } else if (decision === 'PAUSE') {
    gaugeProgress.setAttribute('stroke', '#f59e0b');
    card.classList.add('glow-pause');
    badge.className = 'py-2.5 px-6 rounded-xl font-bold text-center tracking-wider text-sm transition-all bg-amber-500/15 text-amber-400 border border-amber-500/40 shadow-lg shadow-amber-500/10';
    badge.textContent = 'PAUSE — VERIFICATION REQUIRED';
    pulseRing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75';
    pulseDot.className = 'relative inline-flex rounded-full h-3 w-3 bg-amber-500';
    btnOverride.classList.remove('hidden');
    document.getElementById('cardStatusLabel').textContent = 'Awaiting SRE Sign-Off';
  } else {
    gaugeProgress.setAttribute('stroke', '#f43f5e');
    card.classList.add('glow-block');
    badge.className = 'py-2.5 px-6 rounded-xl font-bold text-center tracking-wider text-sm transition-all bg-rose-500/15 text-rose-400 border border-rose-500/40 shadow-lg shadow-rose-500/10';
    badge.textContent = 'BLOCK — ROLLOUT HALTED';
    pulseRing.className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75';
    pulseDot.className = 'relative inline-flex rounded-full h-3 w-3 bg-rose-500';
    btnOverride.classList.add('hidden');
    document.getElementById('cardStatusLabel').textContent = 'Canary Rollout Quarantined';
  }

  // Meta details
  document.getElementById('cardDeployId').textContent = record.deployment_id;
  document.getElementById('detailService').textContent = record.service_name;
  document.getElementById('detailVersion').textContent = record.version;
  document.getElementById('detailCommit').textContent = record.commit_sha;
  document.getElementById('detailEnv').textContent = record.environment;
  document.getElementById('detailDeployer').textContent = record.deployer;
  document.getElementById('detailTimestamp').textContent = record.timestamp;
  document.getElementById('decisionSubtext').textContent = record.status_text;

  // Key factors
  const factorList = document.getElementById('keyFactorsList');
  factorList.innerHTML = '';
  (record.key_factors || []).forEach(f => {
    const li = document.createElement('li');
    li.className = 'flex items-start gap-2';
    const bulletColor = decision === 'ALLOW' ? 'text-emerald-400' : (decision === 'PAUSE' ? 'text-amber-400' : 'text-rose-400');
    li.innerHTML = `<span class="${bulletColor} font-bold">•</span><span>${f}</span>`;
    factorList.appendChild(li);
  });

  // Telemetry tiles
  const tel = record.telemetry || {};
  document.getElementById('telError').textContent = `+${parseFloat(tel.Error_Rate_Delta || 0).toFixed(2)}%`;
  document.getElementById('telLatency').textContent = `${parseFloat(tel.Latency_P95_ms || 50).toFixed(0)}ms`;
  document.getElementById('telLatencyP99').textContent = `${(parseFloat(tel.Latency_P95_ms || 50) + 30).toFixed(0)}ms`;
  document.getElementById('telTests').textContent = `${parseFloat(tel.Unit_Test_Failure_Rate || 0).toFixed(1)}%`;
  document.getElementById('telCVEs').textContent = `${tel.Security_Critical_CVEs || 0} Crit`;
  document.getElementById('telHighCVEs').textContent = tel.Security_High_CVEs || 0;

  // Radar chart update
  if (record.radar_signals) {
    updateRadarChart(record.radar_signals);
  }
}

// Animate Score Counter
function animateScoreCounter(targetScore) {
  const el = document.getElementById('displayScore');
  let current = 0;
  const duration = 750;
  const stepTime = 20;
  const steps = duration / stepTime;
  const increment = targetScore / steps;

  const timer = setInterval(() => {
    current += increment;
    if (current >= targetScore) {
      el.textContent = targetScore;
      clearInterval(timer);
    } else {
      el.textContent = Math.floor(current);
    }
  }, stepTime);
}

// Render Audit Trail Table
function renderAuditTable() {
  const tbody = document.getElementById('auditTableBody');
  tbody.innerHTML = '';

  const filtered = auditRecords.filter(r => currentFilter === 'ALL' || r.decision === currentFilter);

  filtered.forEach(r => {
    const tr = document.createElement('tr');
    tr.className = 'audit-row border-b border-slate-800/40 cursor-pointer';
    tr.onclick = () => renderActiveEvaluation(r);

    const badgeClass = r.decision === 'ALLOW' 
      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' 
      : (r.decision === 'PAUSE' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30' : 'bg-rose-500/10 text-rose-400 border border-rose-500/30');

    tr.innerHTML = `
      <td class="py-3 px-4 font-bold text-cyan-400">${r.deployment_id}</td>
      <td class="py-3 px-4 text-slate-400">${r.timestamp}</td>
      <td class="py-3 px-4 text-white font-semibold">${r.service_name} <span class="text-cyan-400 font-normal">(${r.version})</span></td>
      <td class="py-3 px-4 text-slate-400">${r.commit_sha}</td>
      <td class="py-3 px-4 font-bold ${r.decision === 'ALLOW' ? 'text-emerald-400' : (r.decision === 'PAUSE' ? 'text-amber-400' : 'text-rose-400')}">${r.risk_score}</td>
      <td class="py-3 px-4">
        <span class="px-2.5 py-1 rounded-full text-[10px] font-bold ${badgeClass}">
          ${r.decision}
        </span>
      </td>
      <td class="py-3 px-4 text-slate-400">${r.verified_by || 'auto'}</td>
      <td class="py-3 px-4 text-slate-300 max-w-xs truncate" title="${r.key_factors ? r.key_factors.join('; ') : ''}">${r.key_factors ? r.key_factors[0] : 'Normal metrics'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAudit(filterType) {
  currentFilter = filterType;
  ['All', 'Allow', 'Pause', 'Block'].forEach(f => {
    const btn = document.getElementById(`filter${f}`);
    if (f.toUpperCase() === filterType) {
      btn.className = 'px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40';
    } else {
      btn.className = 'px-3 py-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white border border-slate-700';
    }
  });
  renderAuditTable();
}

// -------------------------------------------------------------
// SIMULATION & API INTEGRATION
// -------------------------------------------------------------
function toggleSimMenu() {
  const menu = document.getElementById('simDropdownMenu');
  menu.classList.toggle('hidden');
}

document.addEventListener('click', (e) => {
  const btn = document.getElementById('simDropdownBtn');
  const menu = document.getElementById('simDropdownMenu');
  if (btn && menu && !btn.contains(e.target) && !menu.contains(e.target)) {
    menu.classList.add('hidden');
  }
});

async function triggerSimulation(scenarioName) {
  toggleSimMenu();
  if (isBackendOnline) {
    try {
      const res = await fetch('/api/simulation/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenarioName })
      });
      const data = await res.json();
      if (data.status === 'success') {
        auditRecords.unshift(data.evaluation);
        renderActiveEvaluation(data.evaluation);
        renderAuditTable();
        computeClientStats();
        return;
      }
    } catch (e) {
      console.warn("Backend simulation error, switching to client-side", e);
    }
  }

  // Client-Side Simulation Scenarios (GitHub Pages)
  const clientScenarios = {
    clean_release: {
      version: `v2.5.${Math.floor(Math.random() * 20)}`,
      commit_sha: Math.random().toString(16).substring(2, 9),
      service_name: "payment-service",
      deployer: "sarah.jenkins",
      telemetry: { Error_Rate_Delta: 0.03, Latency_P95_ms: 54, Unit_Test_Failure_Rate: 0.0, Security_Critical_CVEs: 0, Security_High_CVEs: 0, Auth_Failure_Spike: 0.5, Anomalous_Egress_MB: 18 }
    },
    latency_spike: {
      version: `v2.5.${Math.floor(Math.random() * 20)}`,
      commit_sha: Math.random().toString(16).substring(2, 9),
      service_name: "search-indexer",
      deployer: "alex.chen",
      telemetry: { Error_Rate_Delta: 0.52, Latency_P95_ms: 240, Unit_Test_Failure_Rate: 1.5, Security_Critical_CVEs: 0, Security_High_CVEs: 1, Auth_Failure_Spike: 8.0, Anomalous_Egress_MB: 75 }
    },
    critical_cve_exploit: {
      version: `v2.5.${Math.floor(Math.random() * 20)}`,
      commit_sha: Math.random().toString(16).substring(2, 9),
      service_name: "auth-gateway",
      deployer: "marcus.vance",
      telemetry: { Error_Rate_Delta: 2.1, Latency_P95_ms: 195, Unit_Test_Failure_Rate: 0.0, Security_Critical_CVEs: 2, Security_High_CVEs: 4, Auth_Failure_Spike: 48.0, Anomalous_Egress_MB: 260 }
    },
    massive_failure: {
      version: `v2.5.${Math.floor(Math.random() * 20)}`,
      commit_sha: Math.random().toString(16).substring(2, 9),
      service_name: "order-processing",
      deployer: "ci-bot-auto",
      telemetry: { Error_Rate_Delta: 6.2, Latency_P95_ms: 540, Unit_Test_Failure_Rate: 18.2, Security_Critical_CVEs: 1, Security_High_CVEs: 5, Auth_Failure_Spike: 72.0, Anomalous_Egress_MB: 420 }
    }
  };

  const sc = clientScenarios[scenarioName] || clientScenarios.clean_release;
  const evalResult = clientEvaluateTelemetry(sc.telemetry);

  const newEntry = {
    deployment_id: `DEP-${auditRecords.length + 10001}`,
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
    version: sc.version,
    commit_sha: sc.commit_sha,
    service_name: sc.service_name,
    environment: "canary-prod",
    deployer: sc.deployer,
    risk_score: evalResult.risk_score,
    decision: evalResult.decision,
    color: evalResult.color,
    status_text: evalResult.status_text,
    key_factors: evalResult.key_factors,
    verified_by: evalResult.decision === 'PAUSE' ? 'pending' : 'auto-policy-engine',
    telemetry: sc.telemetry,
    radar_signals: evalResult.radar_signals
  };

  auditRecords.unshift(newEntry);
  renderActiveEvaluation(newEntry);
  renderAuditTable();
  computeClientStats();
}

// Client-Side Calibrated ML Scorer
function clientEvaluateTelemetry(t) {
  const err = parseFloat(t.Error_Rate_Delta || 0);
  const lat = parseFloat(t.Latency_P95_ms || 50);
  const unitFail = parseFloat(t.Unit_Test_Failure_Rate || 0);
  const critCve = parseFloat(t.Security_Critical_CVEs || 0);
  const highCve = parseFloat(t.Security_High_CVEs || 0);
  const authSpike = parseFloat(t.Auth_Failure_Spike || 0);

  let score = (err * 14.0) + (Math.min(lat / 12.0, 25.0)) + (unitFail * 5.5) + (critCve * 28.0) + (highCve * 8.0) + (Math.min(authSpike * 0.4, 15.0));

  if (critCve >= 2) score = Math.max(score, 85);
  else if (critCve === 1) score = Math.max(score, 62);
  if (err >= 4.0) score = Math.max(score, 88);

  const riskScore = Math.min(100, Math.max(0, Math.round(score)));
  let decision = 'ALLOW';
  let color = 'emerald';
  let statusText = 'Deployment Cleared for Production';

  if (riskScore >= 75) {
    decision = 'BLOCK';
    color = 'rose';
    statusText = 'Release Halted - Unsafe Telemetry Detected';
  } else if (riskScore >= 50) {
    decision = 'PAUSE';
    color = 'amber';
    statusText = 'Manual Engineer Verification Required';
  }

  const factors = [];
  if (critCve > 0) factors.push(`${critCve} Critical Security CVE flag(s) present.`);
  if (err > 1.0) factors.push(`Canary error rate delta elevated: +${err.toFixed(2)}%.`);
  if (lat > 200) factors.push(`High P95 latency: ${lat}ms exceeding 200ms threshold.`);
  if (unitFail > 2.0) factors.push(`Unit test failure rate at ${unitFail}%.`);
  if (!factors.length) factors.push('All canary telemetry within optimal operational baselines.');

  return {
    risk_score: riskScore,
    decision: decision,
    color: color,
    status_text: statusText,
    key_factors: factors,
    radar_signals: {
      "Error Rate": Math.min(100, Math.round(err * 15)),
      "Latency": Math.min(100, Math.round(lat / 4.0)),
      "Test Failures": Math.min(100, Math.round(unitFail * 8)),
      "Vulnerabilities": Math.min(100, Math.round(critCve * 35 + highCve * 12)),
      "Network / Auth": Math.min(100, Math.round(authSpike * 2)),
      "Code Blast Radius": 30
    }
  };
}

// -------------------------------------------------------------
// MODALS & ACTIONS
// -------------------------------------------------------------
function openCustomDeployModal() {
  document.getElementById('customDeployModal').classList.remove('hidden');
}

function closeCustomDeployModal() {
  document.getElementById('customDeployModal').classList.add('hidden');
}

async function submitCustomDeploy(e) {
  e.preventDefault();
  const service = document.getElementById('custService').value;
  const version = document.getElementById('custVersion').value;
  const sha = document.getElementById('custSha').value;
  const deployer = document.getElementById('custDeployer').value;

  const telemetry = {
    Error_Rate_Delta: parseFloat(document.getElementById('custError').value),
    Latency_P95_ms: parseFloat(document.getElementById('custLatency').value),
    Latency_P99_ms: parseFloat(document.getElementById('custLatency').value) + 40,
    Security_Critical_CVEs: parseInt(document.getElementById('custCritCVE').value),
    Security_High_CVEs: 0,
    Unit_Test_Failure_Rate: parseFloat(document.getElementById('custUnitTest').value),
    Integration_Test_Passed: 1,
    Anomalous_Egress_MB: 25,
    Auth_Failure_Spike: 1.0,
    Changed_Files_Count: 12,
    Lines_Delta: 400
  };

  closeCustomDeployModal();

  if (isBackendOnline) {
    try {
      const res = await fetch('/api/deployments/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ service_name: service, version, commit_sha: sha, deployer, telemetry })
      });
      const data = await res.json();
      if (data.status === 'success') {
        auditRecords.unshift(data.evaluation);
        renderActiveEvaluation(data.evaluation);
        renderAuditTable();
        computeClientStats();
        return;
      }
    } catch (err) {
      console.warn("Backend eval failed, falling back to client", err);
    }
  }

  // Client evaluate
  const evalResult = clientEvaluateTelemetry(telemetry);
  const newRecord = {
    deployment_id: `DEP-${auditRecords.length + 10001}`,
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
    version,
    commit_sha: sha,
    service_name: service,
    environment: "canary-prod",
    deployer,
    risk_score: evalResult.risk_score,
    decision: evalResult.decision,
    color: evalResult.color,
    status_text: evalResult.status_text,
    key_factors: evalResult.key_factors,
    verified_by: evalResult.decision === 'PAUSE' ? 'pending' : 'auto-policy-engine',
    telemetry,
    radar_signals: evalResult.radar_signals
  };
  auditRecords.unshift(newRecord);
  renderActiveEvaluation(newRecord);
  renderAuditTable();
  computeClientStats();
}

function openOverrideModal() {
  document.getElementById('overrideModal').classList.remove('hidden');
}

function closeOverrideModal() {
  document.getElementById('overrideModal').classList.add('hidden');
}

async function submitOverride(e) {
  e.preventDefault();
  if (!activeEvaluation) return;

  const engineer = document.getElementById('overEngineer').value;
  const action = document.getElementById('overAction').value;
  const reason = document.getElementById('overReason').value;

  closeOverrideModal();

  if (isBackendOnline) {
    try {
      const res = await fetch('/api/deployments/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deployment_id: activeEvaluation.deployment_id,
          action,
          engineer,
          reason
        })
      });
      const data = await res.json();
      if (data.status === 'success') {
        Object.assign(activeEvaluation, data.record);
        renderActiveEvaluation(activeEvaluation);
        renderAuditTable();
        computeClientStats();
        return;
      }
    } catch (err) {
      console.warn("Backend override failed, using client update", err);
    }
  }

  // Client Override
  activeEvaluation.decision = action;
  activeEvaluation.color = action === 'ALLOW' ? 'emerald' : 'rose';
  activeEvaluation.status_text = `Manual Override (${action}): ${reason}`;
  activeEvaluation.verified_by = `${engineer} (Override)`;
  activeEvaluation.key_factors.push(`Manual Override: ${reason}`);

  renderActiveEvaluation(activeEvaluation);
  renderAuditTable();
  computeClientStats();
}

// -------------------------------------------------------------
// CHART.JS INITIALIZATION & UPDATES
// -------------------------------------------------------------
function initCharts() {
  // 1. Radar Chart
  const radarCtx = document.getElementById('radarChart').getContext('2d');
  radarChart = new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: ['Error Rate', 'Latency', 'Test Failures', 'Vulnerabilities', 'Network/Auth', 'Blast Radius'],
      datasets: [{
        label: 'Active Canary Signals',
        data: [15, 20, 10, 5, 10, 20],
        backgroundColor: 'rgba(56, 189, 248, 0.25)',
        borderColor: '#38bdf8',
        borderWidth: 2,
        pointBackgroundColor: '#38bdf8'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: { color: '#94a3b8', font: { size: 10 } },
          ticks: { display: false, max: 100, min: 0 }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // 2. Decision Distribution Doughnut
  const doughnutCtx = document.getElementById('doughnutChart').getContext('2d');
  doughnutChart = new Chart(doughnutCtx, {
    type: 'doughnut',
    data: {
      labels: ['ALLOW', 'PAUSE', 'BLOCK'],
      datasets: [{
        data: [70, 18, 12],
        backgroundColor: ['#10b981', '#f59e0b', '#f43f5e'],
        borderColor: '#0f172a',
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', boxWidth: 10, font: { size: 11 } }
        }
      }
    }
  });

  // 3. Canary Telemetry Timeline Chart
  const timeCtx = document.getElementById('timelineChart').getContext('2d');
  timelineChart = new Chart(timeCtx, {
    type: 'line',
    data: {
      labels: ['T-15m', 'T-12m', 'T-9m', 'T-6m', 'T-3m', 'Canary Live'],
      datasets: [
        {
          label: 'Canary Latency P95 (ms)',
          data: [52, 54, 55, 58, 60, 62],
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          yAxisID: 'y'
        },
        {
          label: 'Error Rate (%)',
          data: [0.02, 0.03, 0.02, 0.04, 0.05, 0.04],
          borderColor: '#f43f5e',
          borderDash: [4, 4],
          borderWidth: 2,
          pointRadius: 3,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { size: 10 } }
        },
        y: {
          type: 'linear',
          position: 'left',
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#38bdf8', font: { size: 10 } }
        },
        y1: {
          type: 'linear',
          position: 'right',
          grid: { display: false },
          ticks: { color: '#f43f5e', font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function updateRadarChart(radarData) {
  if (!radarChart) return;
  const values = [
    radarData['Error Rate'] || 10,
    radarData['Latency'] || 15,
    radarData['Test Failures'] || 5,
    radarData['Vulnerabilities'] || 0,
    radarData['Network / Auth'] || 10,
    radarData['Code Blast Radius'] || 20
  ];
  radarChart.data.datasets[0].data = values;
  radarChart.update();
}

function updateDoughnutChart(allows, pauses, blocks) {
  if (!doughnutChart) return;
  doughnutChart.data.datasets[0].data = [allows, pauses, blocks];
  doughnutChart.update();
}
