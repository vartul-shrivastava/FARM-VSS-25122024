  // Define showLoading and hideLoading in the global scope
  function showLoading() {
    document.getElementById("loadingOverlay").style.display = "flex";
  }

  function hideLoading() {
    document.getElementById("loadingOverlay").style.display = "none";
  }

  // Define setSectionTheme in the global scope
  function setSectionTheme(wrapperId, hasContent) {
    const wrapper = document.getElementById(wrapperId);
    if (!wrapper) return;
    if (hasContent) {
      wrapper.classList.remove('no-content');
      wrapper.classList.add('has-content');
    } else {
      wrapper.classList.remove('has-content');
      wrapper.classList.add('no-content');
    }
  }

  // Function to render dynamic summary content
  function renderDynamicContent(content) {
    const summaryContent = document.getElementById("summaryContent");

    // Clear any existing content
    summaryContent.innerHTML = "";

    if (!content || content.trim() === "") {
      summaryContent.innerHTML = "<p>No summary available.</p>";
      return;
    }

    // Split the content into lines
    const lines = content.trim().split("\n");

    lines.forEach((line) => {
      if (line.startsWith("**") && line.endsWith("**")) {
        // Handle headings
        const headingText = line.replace(/\*\*/g, "").trim();
        const headingElement = document.createElement("h3");
        headingElement.textContent = headingText;
        summaryContent.appendChild(headingElement);
      } else if (line.trim()) {
        // Handle paragraphs and bold text within them
        const paragraphElement = document.createElement("p");

        // Replace **bold** text with <strong> tags
        const formattedText = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        paragraphElement.innerHTML = formattedText;

        summaryContent.appendChild(paragraphElement);
      }
    });
  }

  // Function to show the summary overlay
  function showSummaryOverlay(summaryText) {
    const overlay = document.getElementById('summaryOverlay');
    const summaryContent = document.getElementById('summaryContent');
    
    // Set the summary text
    summaryContent.innerHTML = `<p>${summaryText}</p>`;
    
    // Show the overlay
    overlay.style.display = 'flex';
  }

  // Function to hide the summary overlay
  function hideSummaryOverlay() {
    const overlay = document.getElementById('summaryOverlay');
    overlay.style.display = 'none';
  }
  document.addEventListener("DOMContentLoaded", () => {
const uploadForm = document.getElementById('uploadForm');

uploadForm.addEventListener('submit', function (e) {
  // Prevent default form submission
  e.preventDefault();

  // Clear all dynamically populated content
  clearDynamicContent();

  // Proceed with form submission
  showLoading();
  const formData = new FormData(uploadForm);
  fetch('/ajax_upload', {
    method: 'POST',
    body: formData
  })
    .then(resp => resp.json())
    .then(data => {
      hideLoading();
      if (data.success) {
        document.getElementById('hiddenFilepath').value = data.filepath;
        buildPartitionUI(data.columns, data.auto_ranges, data.chart_paths);
        partitionForm.style.display = 'block';
      } else {
        alert('Error uploading file: ' + data.error);
      }
    })
    .catch(err => {
      hideLoading();
      console.error('Upload Error:', err);
      alert('An error occurred while uploading.');
    });
});

function clearDynamicContent() {
  // Clear processed dataset
  const resultsContainer = document.getElementById('ajaxResultsContainer');
  if (resultsContainer) {
    resultsContainer.innerHTML = "<p>No processed dataset yet.</p>";
    setSectionTheme('processed-dataset-wrapper', false);
  }

  // Clear membership plots
  const plotsContainer = document.getElementById('ajaxPlotsContainer');
  if (plotsContainer) {
    plotsContainer.innerHTML = "<p>No membership function plots yet.</p>";
    setSectionTheme('membership-plots-wrapper', false);
  }

  // Clear step-by-step FARM calculations
  const stepsContainer = document.getElementById('ajaxStepsContainer');
  if (stepsContainer) {
    stepsContainer.innerHTML = "<p>No step-by-step details yet.</p>";
    setSectionTheme('farm-calculations-wrapper', false);
  }

  // Clear FARM rules
  const rulesContainer = document.getElementById('ajaxRulesContainer');
  if (rulesContainer) {
    rulesContainer.innerHTML = "<p>No FARM rules generated yet.</p>";
    setSectionTheme('farm-rules-wrapper', false);
  }

  // Clear explored FARM rules
  const farmSummary = document.getElementById('farmSummary');
  const farmDetails = document.getElementById('farmDetails');
  const farmPlotImage = document.getElementById('farmPlotImage');

  if (farmSummary) farmSummary.innerHTML = "<p>No explored summary available yet.</p>";
  if (farmDetails) farmDetails.innerHTML = "<p>No detailed rules yet.</p>";
  if (farmPlotImage) {
    farmPlotImage.style.display = "none";
    farmPlotImage.src = "";
  }

  console.log("Cleared all dynamically populated content.");
}
});

// Function to handle summary generation
function handleGenerateSummary() {
  if (isGeneratingSummary) return;

  isGeneratingSummary = true;
  const genSummaryBtn = document.getElementById("genSummaryBtn");
  genSummaryBtn.disabled = true;
  showLoading();

  fetch("/generate_summary", { method: "POST" })
      .then((response) => {
          if (!response.ok) {
              throw new Error(`HTTP error! Status: ${response.status}`);
          }
          return response.json();
      })
      .then((data) => {
          hideLoading();
          genSummaryBtn.disabled = false;
          isGeneratingSummary = false;

          if (data.success && data.summary) {
              renderDynamicContent(data.summary);
              document.getElementById("summaryOverlay").style.display = "flex";
          } else {
              renderDynamicContent("No summary available.");
          }
      })
      .catch((error) => {
          hideLoading();
          genSummaryBtn.disabled = false;
          isGeneratingSummary = false;
          console.error("Error fetching summary:", error);
          renderDynamicContent("An error occurred while fetching the summary.");
      });
}

document.addEventListener("DOMContentLoaded", () => {
  // ... existing code ...

  // Handle "Modify Prompt" button click
  modifyPromptBtn.addEventListener('click', () => {
      fetchCurrentPrompt();
      modifyPromptModal.style.display = "flex";
  });

  // Function to fetch the current prompt (default or modified)
  function fetchCurrentPrompt() {
      fetch('/get_current_prompt')
          .then(response => response.json())
          .then(data => {
              if (data.success && data.prompt) {
                  promptTextarea.value = data.prompt;
              } else {
                  promptTextarea.value = 'No prompt available.';
              }
          })
          .catch(err => {
              console.error("Error fetching current prompt:", err);
              promptTextarea.value = 'Error loading prompt.';
          });
  }

  // Function to save the modified prompt
  savePromptBtn.addEventListener('click', () => {
      const modifiedPrompt = promptTextarea.value.trim();
      if (!modifiedPrompt) {
          alert("Prompt cannot be empty.");
          return;
      }

      // Optional: Validate that the prompt contains the {rules} placeholder
      if (!modifiedPrompt.includes("{rules}")) {
          alert("Prompt must include the {rules} placeholder.");
          return;
      }

      showLoading();

      // Send the modified prompt to the backend
      const formData = new FormData();
      formData.append('modified_prompt', modifiedPrompt);

      fetch('/set_modified_prompt', {
          method: 'POST',
          body: formData
      })
          .then(response => response.json())
          .then(data => {
              hideLoading();
              if (data.success) {
                  alert("Prompt modified successfully.");
                  closeModifyPrompt();
              } else {
                  alert(`Error: ${data.error}`);
              }
          })
          .catch(err => {
              hideLoading();
              console.error("Error saving modified prompt:", err);
              alert("An error occurred while saving the modified prompt.");
          });
  });

  // ... existing code ...
});

  let isGeneratingSummary = false;

  document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const uploadForm = document.getElementById('uploadForm');
    const partitionForm = document.getElementById('partitionForm');
    const thresholdForm = document.getElementById('thresholdForm');
    const farmExplorationForm = document.getElementById('farmExplorationForm');
    const setModelBtn = document.getElementById('setModelBtn');
    const modelSelect = document.getElementById('modelSelect');
    const genSummaryBtn = document.getElementById('genSummaryBtn');
    const modifyPromptBtn = document.getElementById('modifyPromptBtn'); // Assuming future use

    // Load available models on page load
    fetch('/get_models')
      .then(response => response.json())
      .then(data => {
        if (data.success && data.models.length > 0) {
          modelSelect.innerHTML = data.models
            .map(model => `<option value="${model}">${model}</option>`)
            .join('');
        } else {
          modelSelect.innerHTML = '<option value="" disabled>No models available</option>';
          console.error("Error fetching models:", data.error || "No models found.");
        }
      })
      .catch(err => {
        modelSelect.innerHTML = '<option value="" disabled>Error loading models</option>';
        console.error("Error fetching models:", err);
      });

    // Handle "Set Model" button click
    setModelBtn.addEventListener('click', () => {
      const selectedModel = modelSelect.value;
      if (!selectedModel) {
        alert('Please select a model.');
        return;
      }

      showLoading();

      // Send the selected model to the backend via AJAX
      const formData = new FormData();
      formData.append('model', selectedModel);

      fetch('/set_model', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          hideLoading();
          if (data.success) {
            alert(`Model "${selectedModel}" set successfully!`);
          } else {
            alert(`Error setting model: ${data.error}`);
          }
        })
        .catch(err => {
          hideLoading();
          console.error("Error setting model:", err);
          alert('An error occurred while setting the model.');
        });
    });

    // Handle "Generate Summary" button click
    genSummaryBtn.addEventListener('click', handleGenerateSummary);

    // Handle "Modify Prompt" button click
    const modifyPromptModal = document.getElementById('modifyPromptModal');
    const closeModifyPromptBtn = document.getElementById('closeModifyPromptBtn');
    const savePromptBtn = document.getElementById('savePromptBtn');
    const cancelPromptBtn = document.getElementById('cancelPromptBtn');
    const resetPromptBtn = document.getElementById('resetPromptBtn');
    const promptTextarea = document.getElementById('promptTextarea');

// Function to fetch the current prompt (default or modified)
function fetchCurrentPrompt() {
  fetch('/get_current_prompt')
      .then(response => response.json())
      .then(data => {
          if (data.success && data.prompt) {
              promptTextarea.value = data.prompt;
          } else {
              promptTextarea.value = 'No prompt available.';
          }
      })
      .catch(err => {
          console.error("Error fetching current prompt:", err);
          promptTextarea.value = 'Error loading prompt.';
      });
}

    // Function to open the Modify Prompt modal
    modifyPromptBtn.addEventListener('click', () => {
      fetchCurrentPrompt();
      modifyPromptModal.style.display = "flex";
    });

    // Function to close the Modify Prompt modal
    function closeModifyPrompt() {
      modifyPromptModal.style.display = "none";
    }

    closeModifyPromptBtn.addEventListener('click', closeModifyPrompt);
    cancelPromptBtn.addEventListener('click', closeModifyPrompt);

// Function to save the modified prompt
savePromptBtn.addEventListener('click', () => {
  const modifiedPrompt = promptTextarea.value.trim();
  if (!modifiedPrompt) {
      alert("Prompt cannot be empty.");
      return;
  }

  // Optional: Validate that the prompt contains the {rules} placeholder
  if (!modifiedPrompt.includes("{rules}")) {
      alert("Prompt must include the {rules} placeholder.");
      return;
  }

  showLoading();

  // Send the modified prompt to the backend
  const formData = new FormData();
  formData.append('modified_prompt', modifiedPrompt);

  fetch('/set_modified_prompt', {
      method: 'POST',
      body: formData
  })
      .then(response => response.json())
      .then(data => {
          hideLoading();
          if (data.success) {
              alert("Prompt modified successfully.");
              closeModifyPrompt();
          } else {
              alert(`Error: ${data.error}`);
          }
      })
      .catch(err => {
          hideLoading();
          console.error("Error saving modified prompt:", err);
          alert("An error occurred while saving the modified prompt.");
      });
});

    // Function to reset the modified prompt to default
    resetPromptBtn.addEventListener('click', () => {
      if (confirm("Are you sure you want to reset the prompt to its default state?")) {
        showLoading();

        fetch('/reset_modified_prompt', {
          method: 'POST'
        })
          .then(response => response.json())
          .then(data => {
            hideLoading();
            if (data.success) {
              alert("Prompt has been reset to default.");
              fetchCurrentPrompt();
            } else {
              alert(`Error: ${data.error}`);
            }
          })
          .catch(err => {
            hideLoading();
            console.error("Error resetting prompt:", err);
            alert("An error occurred while resetting the prompt.");
          });
      }
    });

    // AJAX Upload Form Submission
    uploadForm.addEventListener('submit', function(e) {
      e.preventDefault();
      showLoading();

      const formData = new FormData(uploadForm);
      fetch('/ajax_upload', {
        method: 'POST',
        body: formData
      })
        .then(resp => resp.json())
        .then(data => {
          hideLoading();
          if (data.success) {
            // File successfully uploaded
            document.getElementById('hiddenFilepath').value = data.filepath;
            buildPartitionUI(data.columns, data.auto_ranges, data.chart_paths);
            // Show the partition form
            partitionForm.style.display = 'block';
          } else {
            alert('Error uploading file: ' + data.error);
          }
        })
        .catch(err => {
          hideLoading();
          console.error('Upload Error:', err);
          alert('An error occurred while uploading. Check console for details.');
        });
    });

    // AJAX Partition Form Submission
    partitionForm.addEventListener('submit', function(e) {
      e.preventDefault();
      showLoading();

      const formData = new FormData(partitionForm);
      fetch('/ajax_process', {
        method: 'POST',
        body: formData
      })
        .then(resp => resp.json())
        .then(data => {
          hideLoading();
          if (data.success) {
            // 1) Processed Dataset
            const resultsContainer = document.getElementById('ajaxResultsContainer');
            if (data.results_html && data.results_html.trim()) {
              resultsContainer.innerHTML = data.results_html;
              setSectionTheme('processed-dataset-wrapper', true);
            } else {
              resultsContainer.innerHTML = "<p>No processed dataset yet.</p>";
              setSectionTheme('processed-dataset-wrapper', false);
            }

            // 2) Membership Plots
            const plotsContainer = document.getElementById('ajaxPlotsContainer');
            if (data.plots && data.plots.length > 0) {
              let plotsHTML = "";
              data.plots.forEach(plotFile => {
                plotsHTML += `<img src="/static/plots/${plotFile}" alt="FMF Plot" />`;
              });
              plotsContainer.innerHTML = plotsHTML;
              setSectionTheme('membership-plots-wrapper', true);
            } else {
              plotsContainer.innerHTML = "<p>No membership function plots yet.</p>";
              setSectionTheme('membership-plots-wrapper', false);
            }

            // 3) Step-by-step FARM
            const stepsContainer = document.getElementById('ajaxStepsContainer');
            if (data.steps_html && data.steps_html.trim()) {
              stepsContainer.innerHTML = data.steps_html;
              setSectionTheme('farm-calculations-wrapper', true);
            } else {
              stepsContainer.innerHTML = "<p>No step-by-step details yet.</p>";
              setSectionTheme('farm-calculations-wrapper', false);
            }

            // 4) FARM Rules
            const rulesContainer = document.getElementById('ajaxRulesContainer');
            if (data.rules_html && data.rules_html.trim()) {
              rulesContainer.innerHTML = data.rules_html;
              setSectionTheme('farm-rules-wrapper', true);
            } else {
              rulesContainer.innerHTML = "<p>No FARM rules generated yet.</p>";
              setSectionTheme('farm-rules-wrapper', false);
            }
          } else {
            alert('Error: ' + data.error);
          }
        })
        .catch(err => {
          hideLoading();
          console.error('AJAX Error:', err);
          alert('An error occurred while processing. Check console for details.');
        });
    });

    // FARM Exploration Form Submission
    farmExplorationForm.addEventListener('submit', function(e) {
      e.preventDefault();
      showLoading();

      const formData = new FormData(farmExplorationForm);
      fetch('/explore_farm_rules', {
          method: 'POST',
          body: formData
      })
      .then(resp => {
          if (!resp.ok) {
              throw new Error(`HTTP error! Status: ${resp.status}`);
          }
          return resp.json();
      })
      .then(data => {
          hideLoading();
          if (data.success) {
              // Display Summary
              const farmSummary = document.getElementById('farmSummary');
              const farmDetails = document.getElementById('farmDetails');
              const farmPlotImage = document.getElementById('farmPlotImage');

              farmSummary.innerHTML = `<p>${data.summary}</p>`;
              farmDetails.innerHTML = data.details;
              farmPlotImage.src = `/static/plots/${data.plot_filename}`;
              farmPlotImage.style.display = 'block';

              // Show Results Section
              document.getElementById('farmResults').style.display = 'block';
          } else {
              console.error("Backend returned error:", data.error);
              alert(`Error exploring FARM rules: ${data.error}`);
          }
      })
    });

    // Threshold Form Submission
    thresholdForm.addEventListener('submit', function(e) {
      e.preventDefault();
      showLoading();
      const formData = new FormData(thresholdForm);
      fetch('/set_thresholds', {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          hideLoading();
          if (data.success) {
            alert(`Thresholds set successfully!\n\nMin Support: ${data.min_support}\nMin Certainty: ${data.min_certainty}`);
          } else {
            alert(`Failed to set thresholds: ${data.error}`);
          }
        })
        .catch((error) => {
          hideLoading();
          console.error('Error:', error);
          alert('An error occurred while setting thresholds.');
        });
    });

    // Collapsible Sections Toggle Function
    window.toggleSection = function(sectionId) {
      const content = document.getElementById(`section-${sectionId}`);
      const icon = document.getElementById(`toggle-icon-${sectionId}`);
      if (!content || !icon) return;

      if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
        icon.classList.add('rotate');
      } else {
        content.style.display = 'none';
        icon.classList.remove('rotate');
      }
    };

    // Build Partition UI Function
    window.buildPartitionUI = function(columns, autoRanges, chartPaths) {
      const container = document.getElementById('partitionUI');
      if (!columns || !columns.numerical || columns.numerical.length === 0) {
        container.innerHTML = "<p>No numerical columns found, or none provided yet.</p>";
        return;
      }

      let html = '<div class="config-cards-grid">';

      columns.numerical.forEach(col => {
        const autoMin = autoRanges[col]?.min ?? 0;
        const autoMax = autoRanges[col]?.max ?? 0;

        html += `
          <div class="config-card">
            <h4>${col}</h4>
            <label>
              <input type="checkbox" name="columns" value="${col}" checked />
              Include
            </label>
            <div>
              <label>
                <input
                  type="radio"
                  name="range_type_${col}"
                  value="auto"
                  checked
                  onchange="toggleCustomRangeInputs('${col}', false)"
                />
                Auto Range (${autoMin} - ${autoMax})
              </label>
              <label>
                <input
                  type="radio"
                  name="range_type_${col}"
                  value="manual"
                  onchange="toggleCustomRangeInputs('${col}', true)"
                />
                Custom Range
              </label>
            </div>
        `;

        // Distribution chart if available
        if (chartPaths[col]) {
          html += `
            <img
              src="/static/${chartPaths[col]}"
              alt="Distribution of ${col}"
              class="distribution-chart"
            />
          `;
        }

        html += `
            <div id="custom-range-${col}" class="range-inputs" style="display: none;">
              <label>Min:</label>
              <input type="number" name="min_${col}" placeholder="Min" step="any" />
              <label>Max:</label>
              <input type="number" name="max_${col}" placeholder="Max" step="any" />
            </div>
            <label>Partitions:</label>
            <input
              type="number"
              name="partitions_${col}"
              value="3"
              min="2"
              max="10"
              class="partition-input"
              data-column="${col}"
            />
            <label>
              <input
                type="checkbox"
                name="weight_determine_${col}"
                class="weight-determine"
                data-column="${col}"
              />
              Assign Weights
            </label>
            <div id="weights-container-${col}" class="weights-container" style="display: none;"></div>
          </div>
        `;
      });

      html += '</div> <!-- .config-cards-grid -->';
      container.innerHTML = html;
      rebindPartitionListeners();
    };

    // Handle the close button functionality for Summary Overlay
    document.getElementById("closeSummaryBtn").addEventListener("click", hideSummaryOverlay);
  });

  // Existing helper functions (toggleCustomRangeInputs, rebindPartitionListeners, etc.)

  // Example function for handling file uploads
  function handleFileUpload(e) {
    e.preventDefault();
    showLoading();

    const formData = new FormData(e.target);
    fetch("/ajax_upload", {
      method: "POST",
      body: formData,
    })
      .then((resp) => resp.json())
      .then((data) => {
        hideLoading();
        if (data.success) {
          alert("File uploaded successfully!");
          buildPartitionUI(data.columns, data.auto_ranges, data.chart_paths);
        } else {
          alert(`Error uploading file: ${data.error}`);
        }
      })
      .catch((err) => {
        hideLoading();
        console.error("Upload Error:", err);
        alert("An error occurred during the upload.");
      });
  }

  // Rebind Partition Listeners Function
  function rebindPartitionListeners() {
    // Weight checkbox toggles
    document.querySelectorAll('.weight-determine').forEach(checkbox => {
      checkbox.addEventListener('change', function() {
        const col = this.dataset.column;
        const container = document.getElementById(`weights-container-${col}`);
        const partInput = document.querySelector(`input[name="partitions_${col}"]`);
        const numPartitions = parseInt(partInput.value, 10) || 3;

        if (this.checked) {
          container.innerHTML = '';
          for (let i = 1; i <= numPartitions; i++) {
            const label = document.createElement('label');
            label.textContent = `${col}_P${i}: `;

            const weightField = document.createElement('input');
            weightField.type = 'number';
            weightField.name = `weight_${col}_P${i}`;
            weightField.placeholder = `Weight for ${col}_P${i}`;
            weightField.step = '0.1';
            weightField.min = '0.0';
            weightField.value = '1.0';

            container.appendChild(label);
            container.appendChild(weightField);
            container.appendChild(document.createElement('br'));
          }
          container.style.display = 'block';
        } else {
          container.style.display = 'none';
          container.innerHTML = '';
        }
      });
    });

    // Partition number changes
    document.querySelectorAll('.partition-input').forEach(input => {
      input.addEventListener('change', function() {
        const col = this.dataset.column;
        const container = document.getElementById(`weights-container-${col}`);
        const checkbox = document.querySelector(`input[name="weight_determine_${col}"]`);
        const numPartitions = parseInt(this.value, 10) || 3;

        if (checkbox && checkbox.checked) {
          container.innerHTML = '';
          for (let i = 1; i <= numPartitions; i++) {
            const label = document.createElement('label');
            label.textContent = `${col}_P${i}: `;

            const weightField = document.createElement('input');
            weightField.type = 'number';
            weightField.name = `weight_${col}_P${i}`;
            weightField.placeholder = `Weight for ${col}_P${i}`;
            weightField.step = '0.1';
            weightField.min = '0.0';
            weightField.value = '1.0';

            container.appendChild(label);
            container.appendChild(weightField);
            container.appendChild(document.createElement('br'));
          }
        }
      });
    });
  }

  // Show Summary Overlay Function
  window.showSummaryOverlay = function(summaryText) {
    const overlay = document.getElementById('summaryOverlay');
    const summaryContent = document.getElementById('summaryContent');
    
    // Set the summary text
    summaryContent.innerHTML = `<p>${summaryText}</p>`;
    
    // Show the overlay
    overlay.style.display = 'flex';
  };

  // Hide Summary Overlay Function
  window.hideSummaryOverlay = function() {
    const overlay = document.getElementById('summaryOverlay');
    overlay.style.display = 'none';
  };

  // Attach event listener to the close button of the summary overlay
  const closeSummaryBtn = document.getElementById('closeSummaryBtn');
  closeSummaryBtn.addEventListener('click', hideSummaryOverlay);

  // Optional: Close overlay when clicking outside the content box
  window.addEventListener('click', function(event) {
    const overlay = document.getElementById('summaryOverlay');
    if (event.target == overlay) {
      hideSummaryOverlay();
    }
  });

  // File Input Label Update
  const fileInput = document.getElementById('fileInput');
  const fileLabel = document.querySelector('.custom-file-label');

  fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
      // Display the selected file name
      fileLabel.textContent = fileInput.files[0].name;
    } else {
      // Reset label if no file is selected
      fileLabel.textContent = 'Please select .csv, .xlsx, or .xls file';
    }
  });

  // Function to open specific collapsible sections
  function openCollapsibleSection(sectionId) {
    const content = document.getElementById(`section-${sectionId}`);
    const icon = document.getElementById(`toggle-icon-${sectionId}`);
    
    if (content && icon) {
      content.style.display = 'block'; // Show the content
      icon.classList.add('rotate');    // Rotate the toggle icon
    }
  }

  // Open all collapsible sections by default
  openCollapsibleSection('processed-dataset');
  openCollapsibleSection('membership-plots');
  openCollapsibleSection('farm-calculations');
  openCollapsibleSection('farm-rules');