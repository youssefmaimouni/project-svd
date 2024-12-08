const dragDropArea = document.getElementById("dragDropArea");
const fileInput = document.getElementById("fileInput");
const chooseFileButton = document.getElementById("chooseFileButton");
const output = document.getElementById("output");

// Function to handle files
const handleFiles = (files) => {
    output.innerHTML = ""; // Clear previous images

    for (const file of files) {
        if (!file.type.startsWith("image/")) {
            alert("Only image files are allowed.");
            continue;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement("img");
            img.src = e.target.result;
            img.className = "w-full h-auto rounded-lg shadow-md"; // Tailwind for styling
            output.appendChild(img);
        };
        reader.onerror = (err) => {
            console.error("Error reading file:", err);
            alert("An error occurred while reading the file.");
        };
        reader.readAsDataURL(file);
    }
};

// Drag & Drop area handlers
dragDropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dragDropArea.classList.add("bg-gray-700"); // Visual feedback
});

dragDropArea.addEventListener("dragleave", () => {
    dragDropArea.classList.remove("bg-gray-700");
});

dragDropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDropArea.classList.remove("bg-gray-700");

    const files = e.dataTransfer.files;

    if (files.length === 0) {
        alert("No files selected.");
        return;
    }

    // Update the file input with the dropped files
    const dataTransfer = new DataTransfer();
    for (const file of files) {
        dataTransfer.items.add(file);
    }
    fileInput.files = dataTransfer.files;

    // Process the files
    handleFiles(fileInput.files);
});

// File input handler
fileInput.addEventListener("change", (e) => {
    const files = e.target.files;
    handleFiles(files);
});

// Button click triggers file input
chooseFileButton.addEventListener("click", (e) => {
    e.preventDefault(); // Prevent default action (link behavior)
    fileInput.click();
});
