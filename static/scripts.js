const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;
const undoStack = [];

canvas.addEventListener("mousedown", () => { isDrawing = true; });
canvas.addEventListener("mouseup", () => { isDrawing = false; ctx.beginPath(); undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height)); });
canvas.addEventListener("mousemove", draw);

function draw(event) {
    if (!isDrawing) return;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

document.getElementById("clearButton").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    undoStack.length = 0;
    ctx.beginPath();
});

document.getElementById("undoButton").addEventListener("click", () => {
    if (undoStack.length > 0) {
        ctx.putImageData(undoStack.pop(), 0, 0);
    }
});

document.getElementById("generateButton").addEventListener("click", async () => {
    const canvasData = canvas.toDataURL();
    const prompt = document.getElementById("promptBox").value;
    const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canvas: canvasData, prompt }),
    });
    const data = await response.json();
    document.getElementById("generatedImage").src = data.generated_image;
});

document.getElementById("saveButton").addEventListener("click", async () => {
    const canvasData = canvas.toDataURL();
    const generatedImageSrc = document.getElementById("generatedImage").src;

    if (!generatedImageSrc || generatedImageSrc === "") {
        alert("No generated image to save.");
        return;
    }

    const response = await fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            canvas: canvasData,
            generated: generatedImageSrc,
        }),
    });

    // Trigger file download
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = "sketch_and_generated_image.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
});
