document.addEventListener("DOMContentLoaded", function () {
    // Getting all elements
    var fileInputText = document.getElementById("filePathInput");
    var browseButton = document.getElementById("browseFileBtn");
    var fileChooser = document.getElementById("fileChooser");
    var summaryBox = document.getElementById("summaryTextarea");
    var saveBtn = document.getElementById("saveFileButton");
    var fileNameOutput = document.getElementById("outputFileInput");
    var modeRadios = document.getElementsByName("mode");
    var modelSelectDropdown = document.getElementById("modelDropdownList");

    // Dummy summary text to simulate result
    var demoSummaries = {
        low: "This summary is made with low accuracy. Only main points are shown.",
        medium: "This is a medium level summary with more detailed explanation.",
        high: "This high accuracy summary has detailed information and deep insights."
    };

    // All model options
    var modelOptions = [
        { val: "model1", label: "Model A (Online)", online: true },
        { val: "model2", label: "Model B (Online)", online: true },
        { val: "model3", label: "Model C (Offline)", online: false }
    ];

    // Function to show model list depending on mode
    function refreshModelsBasedOnMode() {
        var selectedMode = document.querySelector('input[name="mode"]:checked').value;
        modelSelectDropdown.innerHTML = "";

        for (var i = 0; i < modelOptions.length; i++) {
            var model = modelOptions[i];
            if (selectedMode === "online" || !model.online) {
                var optionTag = document.createElement("option");
                optionTag.value = model.val;
                optionTag.textContent = model.label;
                modelSelectDropdown.appendChild(optionTag);
            }
        }
    }

    // Add listeners to mode radio buttons
    for (var j = 0; j < modeRadios.length; j++) {
        modeRadios[j].addEventListener("change", refreshModelsBasedOnMode);
    }

    // Set initial model list
    refreshModelsBasedOnMode();

    // Browse button action
    browseButton.addEventListener("click", function () {
        fileChooser.click();
    });

    // File chooser change
    fileChooser.addEventListener("change", function (e) {
        var file = e.target.files[0];
        if (file) {
            fileInputText.value = file.name;
            summaryBox.value = "Reading the PDF and generating summary...";
            // Simulate processing delay
            setTimeout(function () {
                summaryBox.value = demoSummaries["medium"]; // Default summary
            }, 1200);
        }
    });

    // When accuracy changes
    var accuracyRadios = document.getElementsByName("accuracy");
    for (var a = 0; a < accuracyRadios.length; a++) {
        accuracyRadios[a].addEventListener("change", function () {
            var val = document.querySelector('input[name="accuracy"]:checked').value;
            summaryBox.value = demoSummaries[val];
        });
    }

    // Save button logic
    saveBtn.addEventListener("click", function () {
        var content = summaryBox.value;
        var fileName = fileNameOutput.value;

        // This downloads the summary as a text file
        var blob = new Blob([content], { type: "text/plain" });
        var downloadLink = document.createElement("a");
        downloadLink.href = URL.createObjectURL(blob);
        downloadLink.download = fileName;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    });
});
