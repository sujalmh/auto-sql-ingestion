// State management
let currentJobId = null;
let statusPollInterval = null;
let isIncrementalLoad = false;

// DOM Elements
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const previewSection = document.getElementById('preview-section');
const ilPreviewSection = document.getElementById('il-preview-section');
const successSection = document.getElementById('success-section');
const errorSection = document.getElementById('error-section');

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileDescription = document.getElementById('file-description');
const uploadBtn = document.getElementById('upload-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkExistingJob();
});

function setupEventListeners() {
    // File upload
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    uploadBtn.addEventListener('click', uploadFile);

    // OTL Approval
    document.getElementById('approve-btn').addEventListener('click', approveJob);
    document.getElementById('reject-btn').addEventListener('click', rejectJob);

    // IL Approval
    document.getElementById('il-approve-btn').addEventListener('click', approveILJob);
    document.getElementById('il-reject-btn').addEventListener('click', rejectJob);

    // Reset
    document.getElementById('new-upload-btn').addEventListener('click', resetUI);
    document.getElementById('retry-btn').addEventListener('click', resetUI);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        uploadArea.querySelector('p').textContent = `Selected: ${file.name}`;
        uploadBtn.disabled = false;
    }
}

async function uploadFile() {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const description = fileDescription.value.trim();
    if (description) {
        formData.append('file_description', description);
    }

    try {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        currentJobId = data.job_id;
        localStorage.setItem('currentJobId', currentJobId);

        showProcessing();
        startStatusPolling();

    } catch (error) {
        showError(error.message);
    }
}

function showProcessing() {
    uploadSection.style.display = 'none';
    processingSection.style.display = 'block';
    updateProgressStep('upload', 'completed', 'File uploaded');
    updateProgressStep('preprocessing', 'active', 'Processing...');
}

function updateProgressStep(step, status, message) {
    const stepEl = document.querySelector(`[data-step="${step}"]`);
    if (!stepEl) return;

    stepEl.classList.remove('active', 'completed');
    if (status !== 'waiting') {
        stepEl.classList.add(status);
    }
    stepEl.querySelector('.step-status').textContent = message;
}

function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);

    statusPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${currentJobId}`);

            // If job not found (404), clear it and reset UI
            if (response.status === 404) {
                console.log('Job not found, clearing saved job ID');
                stopStatusPolling();
                localStorage.removeItem('currentJobId');
                currentJobId = null;
                resetUI();
                return;
            }

            const data = await response.json();
            handleStatusUpdate(data);

        } catch (error) {
            console.error('Status poll error:', error);
        }
    }, 2000); // Poll every 2 seconds
}

function handleStatusUpdate(data) {
    const status = data.status;

    if (status === 'preprocessing') {
        updateProgressStep('preprocessing', 'active', 'Analyzing file...');
    } else if (status === 'similarity_search') {
        updateProgressStep('preprocessing', 'completed', 'Complete');
        updateProgressStep('similarity_search', 'active', 'Searching for similar tables...');
    } else if (status === 'awaiting_approval') {
        // OTL workflow
        stopStatusPolling();
        updateProgressStep('preprocessing', 'completed', 'Complete');
        updateProgressStep('similarity_search', 'completed', 'No match found');
        updateProgressStep('awaiting_approval', 'active', 'Ready for review');
        isIncrementalLoad = false;
        showPreview(data.preview);
    } else if (status === 'schema_mismatch' || status === 'duplicate_data_detected') {
        // IL workflow
        stopStatusPolling();
        updateProgressStep('preprocessing', 'completed', 'Complete');
        updateProgressStep('similarity_search', 'completed', 'Match found');
        updateProgressStep('awaiting_approval', 'active', 'Ready for review');
        isIncrementalLoad = true;
        showILPreview(data);
    } else if (status === 'approved') {
        updateProgressStep('awaiting_approval', 'completed', 'Approved');
        updateProgressStep('approved', 'active', isIncrementalLoad ? 'Appending to table...' : 'Inserting to database...');
    } else if (status === 'completed' || status === 'incremental_load_completed') {
        stopStatusPolling();
        updateProgressStep('approved', 'completed', 'Complete');
        showSuccess(data.result, isIncrementalLoad);
    } else if (status === 'failed') {
        stopStatusPolling();
        showError(data.error || 'Processing failed');
    }
}

function showPreview(preview) {
    processingSection.style.display = 'none';
    previewSection.style.display = 'block';

    // Basic info - table name is now editable
    document.getElementById('table-name-input').value = preview.proposed_table_name;
    document.getElementById('preview-rows').textContent = preview.total_rows.toLocaleString();
    document.getElementById('preview-columns').textContent = preview.columns.length;

    // LLM metadata
    if (preview.llm_metadata) {
        const llmContent = document.getElementById('llm-metadata-content');
        llmContent.innerHTML = `
            <p><strong>Domain:</strong> ${preview.llm_metadata.suggested_domain}</p>
            <p><strong>Description:</strong> ${preview.llm_metadata.description}</p>
            <p><strong>Period Column:</strong> ${preview.llm_metadata.period_column || 'None detected'}</p>
        `;
    }

    // Sample data table
    renderDataTable(preview.columns, preview.sample_rows, 'preview-table-container');

    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('released-on').value = today;
    document.getElementById('updated-on').value = today;
}

function showILPreview(data) {
    processingSection.style.display = 'none';
    ilPreviewSection.style.display = 'block';

    const ilPreview = data.incremental_load_preview;
    const duplicateDetection = data.duplicate_detection;

    // Show duplicate warning if detected
    if (duplicateDetection && (duplicateDetection.status === 'DUPLICATE' || duplicateDetection.status === 'PARTIAL_OVERLAP')) {
        const warningBox = document.getElementById('duplicate-warning');
        warningBox.style.display = 'flex';
        
        document.getElementById('duplicate-message').textContent = duplicateDetection.message;
        
        if (duplicateDetection.existing_last_value) {
            document.getElementById('existing-period').textContent = `Up to ${duplicateDetection.existing_last_value}`;
        }
        
        if (duplicateDetection.new_first_value && duplicateDetection.new_last_value) {
            document.getElementById('new-period').textContent = `${duplicateDetection.new_first_value} to ${duplicateDetection.new_last_value}`;
        }
    } else {
        document.getElementById('duplicate-warning').style.display = 'none';
    }

    // Matched table info
    const matchedTable = ilPreview.matched_table;
    document.getElementById('il-table-name').textContent = matchedTable.table_name;
    document.getElementById('il-similarity').textContent = `${(matchedTable.similarity_score * 100).toFixed(1)}%`;
    document.getElementById('il-current-rows').textContent = ilPreview.current_rows_count.toLocaleString();
    document.getElementById('il-new-rows').textContent = ilPreview.new_rows_count.toLocaleString();
    document.getElementById('il-total-rows').textContent = ilPreview.total_rows_after.toLocaleString();

    // Schema validation
    const validation = ilPreview.validation_result;
    const statusEl = document.getElementById('validation-status');
    
    if (validation.is_compatible) {
        statusEl.innerHTML = `<div class="status-compatible">✓ Compatible (${validation.match_percentage.toFixed(1)}% match)</div>`;
    } else {
        statusEl.innerHTML = `<div class="status-incompatible">⚠️ Schema Differences Detected (${validation.match_percentage.toFixed(1)}% match)</div>`;
    }

    // Matching columns
    if (validation.matching_columns.length > 0) {
        document.getElementById('matching-columns-section').style.display = 'block';
        document.getElementById('matching-count').textContent = validation.matching_columns.length;
        document.getElementById('matching-columns').innerHTML = validation.matching_columns
            .map(col => `<span class="column-tag column-match">${col}</span>`)
            .join('');
    }

    // Missing columns
    if (validation.missing_columns.length > 0) {
        document.getElementById('missing-columns-section').style.display = 'block';
        document.getElementById('missing-count').textContent = validation.missing_columns.length;
        document.getElementById('missing-columns').innerHTML = validation.missing_columns
            .map(col => `<span class="column-tag column-missing">${col}</span>`)
            .join('');
    }

    // Extra columns
    if (validation.extra_columns.length > 0) {
        document.getElementById('extra-columns-section').style.display = 'block';
        document.getElementById('extra-count').textContent = validation.extra_columns.length;
        document.getElementById('extra-columns').innerHTML = validation.extra_columns
            .map(col => `<span class="column-tag column-extra">${col}</span>`)
            .join('');
    }

    // Sample data preview (from original preview data)
    if (data.preview) {
        renderDataTable(data.preview.columns, data.preview.sample_rows, 'il-preview-table-container');
    }

    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('il-released-on').value = today;
    document.getElementById('il-updated-on').value = today;
}

function renderDataTable(columns, rows, containerId) {
    const container = document.getElementById(containerId);
    const table = document.createElement('table');

    // Headers
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.name;
        th.title = col.type;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Sample rows
    const tbody = document.createElement('tbody');
    rows.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            const value = row[col.name];
            td.textContent = value !== null && value !== undefined ? value : '-';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    container.innerHTML = '';
    container.appendChild(table);
}

async function approveJob() {
    const tableName = document.getElementById('table-name-input').value.trim();
    const source = document.getElementById('source').value.trim();
    const sourceUrl = document.getElementById('source-url').value.trim();
    const releasedOn = document.getElementById('released-on').value;
    const updatedOn = document.getElementById('updated-on').value;
    const businessMetadata = document.getElementById('business-metadata').value.trim();

    if (!tableName || !source || !sourceUrl || !releasedOn || !updatedOn) {
        alert('Please fill in all required fields (including table name)');
        return;
    }

    try {
        document.getElementById('approve-btn').disabled = true;
        document.getElementById('approve-btn').textContent = 'Approving...';

        const formData = new URLSearchParams();
        formData.append('table_name', tableName);
        formData.append('source', source);
        formData.append('source_url', sourceUrl);
        formData.append('released_on', releasedOn + 'T00:00:00');
        formData.append('updated_on', updatedOn + 'T00:00:00');
        if (businessMetadata) {
            formData.append('business_metadata', businessMetadata);
        }

        const response = await fetch(`/approve/${currentJobId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Approval failed');
        }

        // Show processing again and resume polling
        previewSection.style.display = 'none';
        processingSection.style.display = 'block';
        startStatusPolling();

    } catch (error) {
        // Reset button state on error
        document.getElementById('approve-btn').disabled = false;
        document.getElementById('approve-btn').textContent = '✓ Approve & Insert';
        showError(error.message);
    }
}

async function approveILJob() {
    const source = document.getElementById('il-source').value.trim();
    const sourceUrl = document.getElementById('il-source-url').value.trim();
    const releasedOn = document.getElementById('il-released-on').value;
    const updatedOn = document.getElementById('il-updated-on').value;
    const businessMetadata = document.getElementById('il-business-metadata').value.trim();

    if (!source || !sourceUrl || !releasedOn || !updatedOn) {
        alert('Please fill in all required fields');
        return;
    }

    try {
        document.getElementById('il-approve-btn').disabled = true;
        document.getElementById('il-approve-btn').textContent = 'Approving...';

        const formData = new URLSearchParams();
        // For IL, table name comes from matched table (not user input)
        formData.append('source', source);
        formData.append('source_url', sourceUrl);
        formData.append('released_on', releasedOn + 'T00:00:00');
        formData.append('updated_on', updatedOn + 'T00:00:00');
        if (businessMetadata) {
            formData.append('business_metadata', businessMetadata);
        }

        const response = await fetch(`/approve/${currentJobId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Approval failed');
        }

        // Show processing again and resume polling
        ilPreviewSection.style.display = 'none';
        processingSection.style.display = 'block';
        startStatusPolling();

    } catch (error) {
        // Reset button state on error
        document.getElementById('il-approve-btn').disabled = false;
        document.getElementById('il-approve-btn').textContent = '✓ Approve & Append Data';
        showError(error.message);
    }
}

async function rejectJob() {
    if (!confirm('Are you sure you want to reject this job?')) return;

    try {
        const response = await fetch(`/reject/${currentJobId}`, {
            method: 'POST'
        });

        if (response.ok) {
            resetUI();
        }
    } catch (error) {
        showError(error.message);
    }
}

function showSuccess(result, isIL = false) {
    processingSection.style.display = 'none';
    successSection.style.display = 'block';

    // Show load type
    const loadTypeEl = document.getElementById('success-load-type');
    if (isIL) {
        loadTypeEl.innerHTML = '<span class="badge badge-il">🔄 Incremental Load</span>';
        document.getElementById('success-action').textContent = 'Appended';
    } else {
        loadTypeEl.innerHTML = '<span class="badge badge-otl">🆕 One-Time Load</span>';
        document.getElementById('success-action').textContent = 'Inserted';
    }

    document.getElementById('success-table-name').textContent = result.table_name;
    document.getElementById('success-rows').textContent = result.rows_inserted.toLocaleString();

    localStorage.removeItem('currentJobId');
    currentJobId = null;
}

function showError(message) {
    uploadSection.style.display = 'none';
    processingSection.style.display = 'none';
    previewSection.style.display = 'none';
    ilPreviewSection.style.display = 'none';
    successSection.style.display = 'none';
    errorSection.style.display = 'block';

    document.getElementById('error-message').textContent = message;

    stopStatusPolling();
}

function stopStatusPolling() {
    if (statusPollInterval) {
        clearInterval(statusPollInterval);
        statusPollInterval = null;
    }
}

function resetUI() {
    uploadSection.style.display = 'block';
    processingSection.style.display = 'none';
    previewSection.style.display = 'none';
    ilPreviewSection.style.display = 'none';
    successSection.style.display = 'none';
    errorSection.style.display = 'none';

    fileInput.value = '';
    fileDescription.value = '';
    uploadArea.querySelector('p').textContent = 'Drag & drop your file here or click to browse';
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Upload & Process';

    // Reset progress steps
    document.querySelectorAll('.progress-step').forEach(step => {
        step.classList.remove('active', 'completed');
        step.querySelector('.step-status').textContent = 'Waiting...';
    });

    stopStatusPolling();
    localStorage.removeItem('currentJobId');
    currentJobId = null;
    isIncrementalLoad = false;
}

async function checkExistingJob() {
    const savedJobId = localStorage.getItem('currentJobId');
    if (savedJobId) {
        // Verify the job still exists before resuming
        try {
            const response = await fetch(`/status/${savedJobId}`);
            if (response.ok) {
                currentJobId = savedJobId;
                showProcessing();
                startStatusPolling();
            } else {
                // Job doesn't exist anymore, clear it silently
                console.log('Saved job no longer exists, clearing');
                localStorage.removeItem('currentJobId');
                // Don't show error, just let user start fresh
            }
        } catch (error) {
            console.error('Error checking existing job:', error);
            localStorage.removeItem('currentJobId');
            // Don't show error, just let user start fresh
        }
    }
}
