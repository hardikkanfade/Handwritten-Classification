const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const brushSizeInput = document.getElementById('brushSize');
const predictionEl = document.getElementById('prediction');
const confidenceEl = document.getElementById('confidence');
const statusMessage = document.getElementById('statusMessage');
const barsContainer = document.getElementById('barsContainer');

let isDrawing = false;
let predictTimeout = null;

// Initialize the app
function init() {
    // Fill canvas with black (MNIST background is black)
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Setup brush style
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Generate 10 probability bars
    barsContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        barsContainer.innerHTML += `
            <div class="bar-row" id="barRow-${i}">
                <div class="digit-label">${i}</div>
                <div class="bar-track">
                    <div class="bar-fill" id="barFill-${i}"></div>
                </div>
                <div class="prob-label" id="barProb-${i}">0%</div>
            </div>
        `;
    }
}

// Drawing Logic
function startDrawing(e) {
    isDrawing = true;
    if (predictTimeout) clearTimeout(predictTimeout);
    draw(e);
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.beginPath(); // Reset path
    
    // Auto-predict 500ms after drawing stops
    predictTimeout = setTimeout(predict, 500);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault(); // Prevent scrolling on touch devices

    const rect = canvas.getBoundingClientRect();
    let clientX = e.clientX;
    let clientY = e.clientY;

    if (e.touches && e.touches.length > 0) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.lineWidth = brushSizeInput.value;
    ctx.strokeStyle = 'white'; // White brush

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Mouse Events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch Events
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

// Clear Button
clearBtn.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionEl.innerText = '?';
    confidenceEl.innerText = 'Confidence: --%';
    statusMessage.innerText = '';
    statusMessage.className = '';
    resetBars();
});

// Predict Button
predictBtn.addEventListener('click', () => {
    if (predictTimeout) clearTimeout(predictTimeout);
    predict();
});

// Resets visual bar charts
function resetBars() {
    for (let i = 0; i < 10; i++) {
        document.getElementById(`barFill-${i}`).style.width = '0%';
        document.getElementById(`barProb-${i}`).innerText = '0%';
        document.getElementById(`barRow-${i}`).classList.remove('active');
    }
}

// Fetch API for Prediction
async function predict() {
    // Extract base64 PNG data from the canvas
    const dataURL = canvas.toDataURL('image/png');

    statusMessage.innerText = 'Predicting...';
    statusMessage.className = 'loading';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to predict digit.');
        }

        // Update basic text
        predictionEl.innerText = data.digit;
        confidenceEl.innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        statusMessage.innerText = '';
        statusMessage.className = '';

        // Update probability bars
        data.probabilities.forEach((prob, i) => {
            const percent = (prob * 100).toFixed(1);
            document.getElementById(`barFill-${i}`).style.width = `${percent}%`;
            document.getElementById(`barProb-${i}`).innerText = `${Math.round(prob * 100)}%`;

            const row = document.getElementById(`barRow-${i}`);
            if (i === data.digit) {
                row.classList.add('active');
            } else {
                row.classList.remove('active');
            }
        });

    } catch (error) {
        console.error(error);
        statusMessage.innerText = error.message;
        statusMessage.className = '';
        predictionEl.innerText = '?';
        confidenceEl.innerText = 'Confidence: --%';
        resetBars();
    }
}

// Run setup
init();
