// Forest Cover Prediction Frontend JavaScript
// Version: 1.1 - Fixed port configuration

const API_BASE_URL = 'http://localhost:8001';

// Global variables
let modelInfo = null;
let isLoading = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
    setupEventListeners();
    loadModelInfo();
});

// Initialize the application
function initializeApp() {
    console.log('🌲 Forest Cover Prediction System Initialized');
    checkApiHealth();
}

// Setup event listeners
function setupEventListeners() {
    // Prediction form
    document.getElementById('prediction-form').addEventListener('submit', handlePrediction);

    // Load sample data button
    document.getElementById('load-sample').addEventListener('click', loadSampleData);

    // Preset buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', loadPreset);
    });

    // File upload
    document.getElementById('csv-file').addEventListener('change', handleFileUpload);

    // Download template
    document.getElementById('download-template').addEventListener('click', downloadTemplate);

    // Drag and drop for file upload
    setupDragAndDrop();
}

// Check API health
async function checkApiHealth() {
    try {
        console.log(`🌐 Attempting to connect to: ${API_BASE_URL}/health`);
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            updateModelStatus('Ready', 'success');
            document.getElementById('device-info').textContent = data.cuda_available ? 'CUDA' : 'CPU';
            document.getElementById('device-info').className = data.cuda_available ? 'badge bg-success' : 'badge bg-info';
        } else {
            updateModelStatus('Error', 'danger');
        }
    } catch (error) {
        console.error('API Health Check Failed:', error);
        updateModelStatus('Offline', 'danger');
    }
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        modelInfo = await response.json();

        // Update accuracy display
        document.getElementById('model-accuracy').textContent = `${(modelInfo.accuracy * 100).toFixed(1)}%`;

        // Populate cover types information
        populateCoverTypesInfo();

    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

// Update model status
function updateModelStatus(status, type) {
    const statusElement = document.getElementById('model-status');
    statusElement.textContent = status;
    statusElement.className = `badge bg-${type}`;
}

// Handle prediction form submission
async function handlePrediction(event) {
    event.preventDefault();

    if (isLoading) return;

    isLoading = true;
    showLoadingModal();

    try {
        const formData = new FormData(event.target);
        const predictionData = buildPredictionData(formData);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayPredictionResult(result);

    } catch (error) {
        console.error('Prediction failed:', error);
        showError('Prediction failed. Please check your input and try again.');
    } finally {
        isLoading = false;
        hideLoadingModal();
    }
}

// Build prediction data from form
function buildPredictionData(formData) {
    const data = {
        elevation: parseFloat(formData.get('elevation')),
        aspect: parseFloat(formData.get('aspect')),
        slope: parseFloat(formData.get('slope')),
        horizontal_distance_to_hydrology: parseFloat(formData.get('horizontal_distance_to_hydrology')),
        vertical_distance_to_hydrology: parseFloat(formData.get('vertical_distance_to_hydrology')),
        horizontal_distance_to_roadways: parseFloat(formData.get('horizontal_distance_to_roadways')),
        hillshade_9am: parseFloat(formData.get('hillshade_9am')),
        hillshade_noon: parseFloat(formData.get('hillshade_noon')),
        hillshade_3pm: parseFloat(formData.get('hillshade_3pm')),
        horizontal_distance_to_fire_points: parseFloat(formData.get('horizontal_distance_to_fire_points'))
    };

    // Initialize all wilderness areas to 0
    for (let i = 1; i <= 4; i++) {
        data[`wilderness_area${i}`] = 0;
    }

    // Set selected wilderness area to 1
    const wildernessArea = formData.get('wilderness_area');
    if (wildernessArea) {
        data[`wilderness_area${wildernessArea}`] = 1;
    }

    // Initialize all soil types to 0
    for (let i = 1; i <= 40; i++) {
        data[`soil_type${i}`] = 0;
    }

    // Set selected soil type to 1
    const soilType = formData.get('soil_type');
    if (soilType) {
        data[`soil_type${soilType}`] = 1;
    }

    return data;
}

// Display prediction result
function displayPredictionResult(result) {
    const resultsPanel = document.getElementById('results-panel');

    const confidencePercentage = (result.confidence * 100).toFixed(1);
    const executionTime = (result.execution_time * 1000).toFixed(0);

    resultsPanel.innerHTML = `
        <div class="result-card fade-in">
            <div class="result-prediction cover-type-${result.prediction}">
                Type ${result.prediction}
            </div>
            <div class="result-description">
                ${result.description}
            </div>
            <div class="result-confidence">
                <strong>Confidence: ${confidencePercentage}%</strong>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercentage}%"></div>
                </div>
            </div>
            <div class="row text-start">
                <div class="col-6">
                    <small><strong>Elevation Zone:</strong><br>${result.elevation_zone}</small>
                </div>
                <div class="col-6">
                    <small><strong>Terrain:</strong><br>${result.terrain}</small>
                </div>
            </div>
            <div class="reasoning-steps">
                <h6><i class="fas fa-brain"></i> AI Reasoning Steps:</h6>
                <ol>
                    ${result.reasoning.map(step => `<li>${step}</li>`).join('')}
                </ol>
            </div>
            <div class="mt-3">
                <small class="text-light">
                    <i class="fas fa-clock"></i> Processed in ${executionTime}ms
                    | <i class="fas fa-cog"></i> ${result.model_used} model
                </small>
            </div>
        </div>
    `;
}

// Load sample data
function loadSampleData() {
    const sampleData = {
        elevation: 3200,
        aspect: 215,
        slope: 18,
        horizontal_distance_to_hydrology: 450,
        vertical_distance_to_hydrology: 75,
        horizontal_distance_to_roadways: 1200,
        hillshade_9am: 195,
        hillshade_noon: 235,
        hillshade_3pm: 158,
        horizontal_distance_to_fire_points: 2100,
        wilderness_area: 3,
        soil_type: 10
    };

    populateForm(sampleData);
    showSuccess('Sample data loaded successfully!');
}

// Load preset data
function loadPreset(event) {
    const preset = event.target.dataset.preset;
    let presetData = {};

    switch (preset) {
        case 'high-elevation':
            presetData = {
                elevation: 3800,
                aspect: 180,
                slope: 25,
                horizontal_distance_to_hydrology: 800,
                vertical_distance_to_hydrology: 150,
                horizontal_distance_to_roadways: 2500,
                hillshade_9am: 180,
                hillshade_noon: 210,
                hillshade_3pm: 130,
                horizontal_distance_to_fire_points: 3000,
                wilderness_area: 1,
                soil_type: 7
            };
            break;
        case 'low-elevation':
            presetData = {
                elevation: 2200,
                aspect: 90,
                slope: 8,
                horizontal_distance_to_hydrology: 200,
                vertical_distance_to_hydrology: 20,
                horizontal_distance_to_roadways: 800,
                hillshade_9am: 220,
                hillshade_noon: 250,
                hillshade_3pm: 180,
                horizontal_distance_to_fire_points: 1500,
                wilderness_area: 2,
                soil_type: 12
            };
            break;
        case 'riparian':
            presetData = {
                elevation: 2600,
                aspect: 270,
                slope: 5,
                horizontal_distance_to_hydrology: 30,
                vertical_distance_to_hydrology: -20,
                horizontal_distance_to_roadways: 1500,
                hillshade_9am: 200,
                hillshade_noon: 240,
                hillshade_3pm: 160,
                horizontal_distance_to_fire_points: 2200,
                wilderness_area: 4,
                soil_type: 15
            };
            break;
        case 'dry-slope':
            presetData = {
                elevation: 2900,
                aspect: 225,
                slope: 30,
                horizontal_distance_to_hydrology: 1200,
                vertical_distance_to_hydrology: 200,
                horizontal_distance_to_roadways: 600,
                hillshade_9am: 160,
                hillshade_noon: 190,
                hillshade_3pm: 120,
                horizontal_distance_to_fire_points: 800,
                wilderness_area: 3,
                soil_type: 8
            };
            break;
    }

    populateForm(presetData);
    showSuccess(`${preset.replace('-', ' ')} preset loaded!`);
}

// Populate form with data
function populateForm(data) {
    Object.keys(data).forEach(key => {
        const element = document.querySelector(`[name="${key}"]`);
        if (element) {
            if (element.type === 'radio') {
                const radioBtn = document.querySelector(`[name="${key}"][value="${data[key]}"]`);
                if (radioBtn) radioBtn.checked = true;
            } else {
                element.value = data[key];
            }
        }
    });
}

// Handle file upload for batch prediction
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
        showError('Please select a CSV file.');
        return;
    }

    if (file.size > 5 * 1024 * 1024) { // 5MB limit
        showError('File size must be less than 5MB.');
        return;
    }

    showLoadingModal();

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/predict/file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayBatchResults(result.predictions);

    } catch (error) {
        console.error('Batch prediction failed:', error);
        showError('Batch prediction failed. Please check your file format.');
    } finally {
        hideLoadingModal();
    }
}

// Display batch prediction results
function displayBatchResults(predictions) {
    const batchResults = document.getElementById('batch-results');
    const tableBody = document.querySelector('#batch-results-table tbody');

    tableBody.innerHTML = '';

    predictions.forEach((result, index) => {
        if (result.error) {
            tableBody.innerHTML += `
                <tr>
                    <td>${index + 1}</td>
                    <td colspan="3" class="text-danger">Error: ${result.error}</td>
                </tr>
            `;
        } else {
            const confidencePercentage = (result.confidence * 100).toFixed(1);
            tableBody.innerHTML += `
                <tr>
                    <td>${index + 1}</td>
                    <td><span class="badge cover-type-${result.prediction}">Type ${result.prediction}</span></td>
                    <td>${confidencePercentage}%</td>
                    <td>${result.description}</td>
                </tr>
            `;
        }
    });

    batchResults.style.display = 'block';
    batchResults.scrollIntoView({ behavior: 'smooth' });
}

// Setup drag and drop for file upload
function setupDragAndDrop() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('csv-file');

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload({ target: { files } });
        }
    });
}

// Download CSV template
function downloadTemplate() {
    const headers = [
        'elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
        'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',
        'hillshade_9am', 'hillshade_noon', 'hillshade_3pm',
        'horizontal_distance_to_fire_points'
    ];

    // Add wilderness areas
    for (let i = 1; i <= 4; i++) {
        headers.push(`wilderness_area${i}`);
    }

    // Add soil types
    for (let i = 1; i <= 40; i++) {
        headers.push(`soil_type${i}`);
    }

    // Sample data row
    const sampleRow = [
        3200, 215, 18, 450, 75, 1200, 195, 235, 158, 2100,
        0, 0, 1, 0, // wilderness areas
        ...Array(40).fill(0) // soil types
    ];
    sampleRow[13] = 1; // Set soil_type_10 to 1

    const csvContent = [
        headers.join(','),
        sampleRow.join(',')
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'forest_cover_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showSuccess('Template downloaded successfully!');
}

// Populate cover types information
function populateCoverTypesInfo() {
    if (!modelInfo || !modelInfo.descriptions) return;

    const coverTypesDiv = document.getElementById('cover-types-info');
    let html = '';

    Object.entries(modelInfo.descriptions).forEach(([type, description]) => {
        html += `
            <div class="cover-type-badge cover-type-${type}">
                <strong>Type ${type}:</strong> ${description}
            </div>
        `;
    });

    coverTypesDiv.innerHTML = html;
}

// Utility functions for UI feedback
function showLoadingModal() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoadingModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) modal.hide();
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showError(message) {
    showToast(message, 'danger');
}

function showToast(message, type) {
    // Create toast element
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    // Add toast to container
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    // Show toast
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
