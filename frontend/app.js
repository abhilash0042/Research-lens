let chatHistory = [];

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    fetchPapers();
    
    // Setup File Upload
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');

    dropArea.addEventListener('click', () => fileInput.click());

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = 'var(--accent-color)';
        dropArea.style.background = 'var(--glass-hover)';
    });

    dropArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = 'var(--glass-border)';
        dropArea.style.background = 'transparent';
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.style.borderColor = 'var(--glass-border)';
        dropArea.style.background = 'transparent';
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    });

    // Chat input enter key
    document.getElementById('chat-input').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

// --- Tab Navigation ---
function switchTab(tabId, event) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    event.currentTarget.classList.add('active');
    document.getElementById(`tab-${tabId}`).classList.add('active');
}

// --- Loading Overlay ---
function showLoader(text) {
    document.getElementById('loader-text').innerText = text;
    document.getElementById('loader').style.display = 'flex';
}
function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

// --- Citation Formatting ---
function formatCitations(text) {
    // Convert [SOURCE 1: Title, Section] into styled spans
    return text.replace(/(\[SOURCE \d+:.*?\])/g, '<span class="citation" title="View Source">$1</span>');
}

// --- API Calls ---

async function fetchPapers() {
    try {
        const res = await fetch('/api/papers');
        const data = await res.json();
        
        const list = document.getElementById('papers-list');
        const selector = document.getElementById('summary-selector');
        
        if (data.papers.length === 0) {
            list.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem; font-style: italic;">No papers loaded.</div>';
            selector.innerHTML = '<option value="">Select a paper...</option>';
            return;
        }

        list.innerHTML = '';
        selector.innerHTML = '<option value="">Select a paper...</option>';

        data.papers.forEach(p => {
            list.innerHTML += `<div class="paper-item"><strong>${p.title}</strong><br><span style="color:var(--text-muted); font-size:0.8rem;">Year: ${p.year}</span></div>`;
            selector.innerHTML += `<option value="${p.paper_id}">${p.title}</option>`;
        });
    } catch (e) {
        console.error("Failed to fetch papers", e);
    }
}

async function handleFiles(files) {
    for (let file of files) {
        if (file.type !== "application/pdf") {
            alert("Only PDFs are supported.");
            continue;
        }

        const formData = new FormData();
        formData.append('file', file);

        showLoader(`Ingesting ${file.name}...`);
        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail);
            
            // Refresh list
            await fetchPapers();
        } catch (e) {
            alert(`Error uploading ${file.name}: ${e.message}`);
        }
    }
    hideLoader();
}

async function clearMemory() {
    await fetch('/api/clear', { method: 'POST' });
    chatHistory = [];
    document.getElementById('chat-window').innerHTML = '<div class="message assistant">Memory cleared. Upload new papers to begin.</div>';
    document.getElementById('summary-result').style.display = 'none';
    document.getElementById('intel-result').style.display = 'none';
    await fetchPapers();
}

// --- Chat ---
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    const chatWindow = document.getElementById('chat-window');

    // Add User Message
    chatWindow.innerHTML += `<div class="message user">${text}</div>`;
    chatWindow.scrollTop = chatWindow.scrollHeight;

    showLoader("Searching and Reasoning...");
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, history: chatHistory })
        });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail);

        // Update history
        chatHistory.push({ role: "user", content: text });
        chatHistory.push({ role: "assistant", content: data.answer });

        // Add Assistant Message
        // Use marked to parse markdown, then format citations
        let parsed = marked.parse(data.answer);
        parsed = formatCitations(parsed);
        
        chatWindow.innerHTML += `<div class="message assistant">${parsed}</div>`;
        chatWindow.scrollTop = chatWindow.scrollHeight;

    } catch (e) {
        chatWindow.innerHTML += `<div class="message assistant" style="color: #ff7675;">Error: ${e.message}</div>`;
    }
    
    hideLoader();
}

// --- Summaries ---
async function generateSummary() {
    const selector = document.getElementById('summary-selector');
    const paperId = selector.value;
    if (!paperId) {
        alert("Please select a paper first.");
        return;
    }

    showLoader("Generating Deep Structured Summary...");
    const resultDiv = document.getElementById('summary-result');
    resultDiv.style.display = 'none';

    try {
        const res = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_id: paperId })
        });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail);

        resultDiv.innerHTML = `
            <h3 style="color: var(--accent-color);">${data.title}</h3>
            
            <div style="margin-top: 16px;">
                <strong style="color: #a8c0ff;">🎯 Core Contribution</strong>
                <p style="margin-top: 8px;">${data.contribution}</p>
            </div>
            
            <div style="margin-top: 16px;">
                <strong style="color: #a8c0ff;">🔬 Methodology</strong>
                <p style="margin-top: 8px;">${data.methodology}</p>
            </div>
            
            <div style="margin-top: 16px;">
                <strong style="color: #a8c0ff;">📊 Results</strong>
                <p style="margin-top: 8px;">${data.results}</p>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px;">
                <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                    <strong style="color: #a8c0ff;">Datasets Used</strong>
                    <p style="font-size: 0.9rem; margin-top: 8px;">${data.datasets}</p>
                </div>
                <div style="background: rgba(255,118,117,0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,118,117,0.3);">
                    <strong style="color: #ff7675;">Limitations & Future Work</strong>
                    <p style="font-size: 0.9rem; margin-top: 8px;">${data.limitations}</p>
                </div>
            </div>
        `;
        resultDiv.style.display = 'block';

    } catch (e) {
        alert("Error: " + e.message);
    }
    
    hideLoader();
}

// --- Intelligence ---
async function runIntelligence(action) {
    const actionNames = {
        'compare': 'Analyzing dimensions...',
        'contradictions': 'Detecting conflicting claims...',
        'review': 'Synthesizing multi-cited review...',
        'hypotheses': 'Finding gaps and generating novel research ideas...'
    };

    showLoader(actionNames[action]);
    const resultDiv = document.getElementById('intel-result');
    resultDiv.style.display = 'none';

    try {
        const res = await fetch('/api/intelligence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail);

        let closeBtnHtml = `<div style="text-align: right; margin-bottom: 10px;"><button onclick="document.getElementById('intel-result').style.display='none'" style="background: transparent; color: var(--text-muted); border: 1px solid var(--glass-border); padding: 6px 12px; border-radius: 6px; cursor: pointer;"><i class="fa-solid fa-times"></i> Close</button></div>`;
        resultDiv.innerHTML = closeBtnHtml;

        if (data.type === 'table') {
            let html = '<h3 style="color: var(--accent-color);"><i class="fa-solid fa-table"></i> Comparison Matrix</h3>';
            html += '<table style="width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.9rem;">';
            
            if (data.data.length > 0) {
                const firstRowValues = data.data[0].values;
                const paperTitles = Array.isArray(firstRowValues) ? [] : Object.keys(firstRowValues);
                
                html += '<thead><tr><th style="border: 1px solid var(--glass-border); padding: 12px; text-align: left; background: rgba(0,0,0,0.4);">Dimension</th>';
                paperTitles.forEach(title => {
                    html += `<th style="border: 1px solid var(--glass-border); padding: 12px; text-align: left; background: rgba(0,0,0,0.4);">${title}</th>`;
                });
                html += '</tr></thead>';
            }
            
            html += '<tbody>';
            data.data.forEach(row => {
                html += '<tr>';
                html += `<td style="border: 1px solid var(--glass-border); padding: 12px; font-weight: bold; background: rgba(0,0,0,0.2); width: 150px;">${row.dimension}</td>`;
                const values = Array.isArray(row.values) ? row.values : Object.values(row.values);
                values.forEach(val => {
                    html += `<td style="border: 1px solid var(--glass-border); padding: 12px;">${val}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            resultDiv.innerHTML += html;
        }
        else if (data.type === 'contradictions') {
            let html = '<h3 style="color: var(--accent-color);"><i class="fa-solid fa-bolt"></i> Conflicting Claims Found</h3>';
            if (data.data.length === 0) {
                html += '<p style="color: #55efc4; margin-top: 12px;">No major contradictions found among the papers.</p>';
            } else {
                data.data.forEach((c, i) => {
                    html += `
                    <div style="background: rgba(255,118,117,0.1); border: 1px solid rgba(255,118,117,0.3); border-radius: 8px; padding: 16px; margin-top: 16px;">
                        <h4 style="color: #ff7675; margin-bottom: 12px;">Contradiction ${i+1}: ${c.paper_a} vs ${c.paper_b}</h4>
                        <p style="font-size: 0.9rem;"><strong>${c.paper_a} claims:</strong> ${c.claim_a}</p>
                        <p style="font-size: 0.9rem; margin-top: 8px;"><strong>${c.paper_b} claims:</strong> ${c.claim_b}</p>
                        <p style="font-size: 0.9rem; margin-top: 12px; color: #a8c0ff;"><strong>AI Analysis:</strong> ${c.explanation}</p>
                    </div>`;
                });
            }
            resultDiv.innerHTML += html;
        }
        else if (data.type === 'text') {
            let title = action === 'review' ? 'Synthesized Literature Review' : 'Novel Research Hypotheses';
            let parsed = marked.parse(data.data);
            parsed = formatCitations(parsed);
            resultDiv.innerHTML += `<h3 style="color: var(--accent-color); margin-bottom: 16px;">${title}</h3>${parsed}`;
        }

        resultDiv.style.display = 'block';
        setTimeout(() => {
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

    } catch (e) {
        alert("Error: " + e.message);
    }
    
    hideLoader();
}
