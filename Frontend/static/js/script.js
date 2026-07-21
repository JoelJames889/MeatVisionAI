const dropArea = document.getElementById("dropArea");
const input = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const form = document.getElementById("uploadForm");
const button = document.querySelector(".analyze-btn");

dropArea.addEventListener("click", () => {
    input.click();
});

input.addEventListener("change", () => {
    if (input.files.length > 0) {
        fileName.innerHTML = input.files[0].name;
    }
});

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.borderColor = "#22c55e";
    dropArea.style.background = "#1e293b";
});

dropArea.addEventListener("dragleave", () => {
    dropArea.style.borderColor = "#3b82f6";
    dropArea.style.background = "transparent";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();

    dropArea.style.borderColor = "#3b82f6";
    dropArea.style.background = "transparent";

    input.files = e.dataTransfer.files;

    if (input.files.length > 0) {
        fileName.innerHTML = input.files[0].name;
    }
});

form.addEventListener("submit", () => {

    button.innerHTML = "Analyzing...";

    button.disabled = true;

});