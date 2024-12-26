import os
import re
import itertools
import uuid
import json
import logging
from datetime import timedelta
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import ollama  # The Ollama Python library for AI models

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    jsonify, session, send_from_directory
)
from werkzeug.utils import secure_filename
from flask_session import Session

# ------------------------------------------------------
# Matplotlib Configuration (headless)
# ------------------------------------------------------
plt.switch_backend('Agg')

# ------------------------------------------------------
# Flask Setup
# ------------------------------------------------------
app = Flask(__name__)
# Ensure necessary directories exist


# Load configuration from environment variables or set defaults
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_secret_key')  # Replace with a secure key
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['PLOTS_FOLDER'] = os.environ.get('PLOTS_FOLDER', 'static/plots')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Flask-Session Configuration
app.config['SESSION_TYPE'] = 'filesystem'  # Use 'redis' or other types in production
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'flask_session')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True  # Adds an extra layer of security
app.config['SESSION_KEY_PREFIX'] = 'farm_vss_session:'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
# Initialize Flask-Session
Session(app)

# ------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------
def allowed_file(filename):
    """
    Check if uploaded file has .csv, .xls, or .xlsx extension.
    """
    session['min_support'] = 0.15
    session['min_certainty'] = 0.60
    valid_extensions = {'csv', 'xls', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in valid_extensions

def get_columns(filepath):
    """
    Identify numeric and categorical columns in a CSV or Excel file.
    Returns {'numerical': [...], 'categorical': [...]}
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        logger.debug(f"Identified columns - Numerical: {numeric_cols}, Categorical: {categorical_cols}")
        return {'numerical': numeric_cols, 'categorical': categorical_cols}
    except Exception as e:
        logger.error(f"Error in get_columns: {e}")

import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def generate_distribution_charts(df, folder='static/plots'):
    """
    Generate visually enhanced frequency distribution charts with KDE for each numeric column in `df`.
    Saves them in `folder` and returns {col: chart_path}.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        chart_paths = {}
        numeric_cols = df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            safe_col = re.sub(r'[^\w\s]', '_', col).replace(' ', '_')
            plot_filename = f"{safe_col}_freq_kde.png"
            plot_path = os.path.join(folder, plot_filename)

            # Close any previous plots
            plt.close('all')

            # Set a beautiful seaborn style
            sns.set_theme(style="whitegrid")

            # Create the figure
            plt.figure(figsize=(10, 6))

            # Generate the frequency distribution with KDE
            sns.histplot(
                data=df,
                x=col,
                kde=True,
                color="#4c72b0",
                stat="density",
                alpha=0.7,
                edgecolor="black"
            )

            # Add detailed labels and title
            plt.title(
                f"Frequency Distribution with KDE of {col}",
                fontsize=16,
                fontweight="bold",
                color="#333333",
            )
            plt.xlabel(col, fontsize=14, fontweight="medium")
            plt.ylabel("Density", fontsize=14, fontweight="medium")
            plt.grid(
                which="major",
                linestyle="--",
                linewidth=0.6,
                color="gray",
                alpha=0.7,
            )

            # Adjust layout
            plt.tight_layout()

            # Save the plot
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close('all')  # Close the current plot to free resources

            # Store the plot path
            chart_paths[col] = f"plots/{plot_filename}"
            logger.debug(f"Generated frequency distribution chart with KDE for {col}")

        return chart_paths
    except Exception as e:
        logger.error(f"Error in generate_distribution_charts: {e}")
        raise



def calculate_membership(x, centers, index):
    """
    Triangular membership. If index=0 => saturates lower bound,
    if last => saturates upper bound, else linear interpolation.
    """
    try:
        if index == 0:
            if x <= centers[index]:
                return 1
            elif x <= centers[index + 1]:
                return (centers[index + 1] - x) / (centers[index + 1] - centers[index])
            else:
                return 0
        elif index == len(centers) - 1:
            if x >= centers[index]:
                return 1
            elif x >= centers[index - 1]:
                return (x - centers[index - 1]) / (centers[index] - centers[index - 1])
            else:
                return 0
        else:
            if x <= centers[index - 1] or x >= centers[index + 1]:
                return 0
            elif x <= centers[index]:
                return (x - centers[index - 1]) / (centers[index] - centers[index - 1])
            else:
                return (centers[index + 1] - x) / (centers[index + 1] - centers[index])
    except Exception as e:
        logger.error(f"Error in calculate_membership: {e}")
        return 0

def apply_fuzzy_logic(filepath, col_partitions, col_ranges, partition_weights):
    """
    Creates fuzzy membership columns for each chosen column in the dataset.
    Generates membership plots. Returns (DataFrame, list_of_plot_filenames).
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        df = df[list(col_partitions.keys())]  # keep only selected cols

        fuzzy_df = pd.DataFrame(index=df.index)
        generated_plots = []

        for col, num_parts in col_partitions.items():
            user_min, user_max = col_ranges[col]
            actual_min = float(user_min) if user_min is not None else float(df[col].min())
            actual_max = float(user_max) if user_max is not None else float(df[col].max())

            if num_parts > 1:
                interval = (actual_max - actual_min) / (num_parts - 1)
                centers = [actual_min + i * interval for i in range(num_parts)]
            else:
                centers = [actual_min]

            # Membership columns
            for i in range(num_parts):
                part_col = f"{col}_P{i+1}"
                fuzzy_df[part_col] = df[col].apply(lambda val: calculate_membership(val, centers, i))

            # Plot membership
            x_vals = np.linspace(actual_min, actual_max, 500)
            plt.figure(figsize=(8, 6))
            for i in range(num_parts):
                y_vals = [calculate_membership(x, centers, i) for x in x_vals]
                plt.plot(x_vals, y_vals, label=f"{col}_P{i+1}")
            safe_name = re.sub(r'[^\w\s-]', '_', col).replace(' ', '_')
            filename = f"{safe_name}_fmf.png"
            full_path = os.path.join(app.config['PLOTS_FOLDER'], filename)

            plt.title(f"Fuzzy Membership for {col}")
            plt.xlabel("Value Range")
            plt.ylabel("Membership Degree")
            plt.legend()
            plt.grid(True)
            plt.savefig(full_path)
            plt.close()

            generated_plots.append(filename)
            logger.debug(f"Generated fuzzy membership plot for {col}: {filename}")

        return fuzzy_df, generated_plots
    except Exception as e:
        logger.error(f"Error in apply_fuzzy_logic: {e}")
        raise

def calculate_significance(combo, df, weights, min_support=0.2):
    """
    For a combo of columns (e.g., col_P1, col_P2), if membership*weight < min_support => 0 for that row.
    Else multiply in. Final significance = average of row products.
    """
    try:
        total = len(df)
        sum_sig = 0.0

        for _, row in df.iterrows():
            product = 1.0
            for col_partition in combo:
                membership_val = float(row[col_partition])
                base_col = col_partition.split("_P")[0]
                weight = float(weights.get(base_col, {}).get(col_partition, 1.0))
                if membership_val * weight >= min_support:
                    product *= (membership_val * weight)
                else:
                    product = 0
                    break
            sum_sig += product

        significance = (sum_sig / total) if total else 0.0
        return significance
    except Exception as e:
        logger.error(f"Error in calculate_significance: {e}")
        return 0.0

def determine_farm_rules(df, weights, min_support=0.2):
    """
    Multi-phase itemset generation until no combos remain.
    Only combos from distinct base columns. Return (frequent_itemsets, step_details).
    """
    try:
        cols = df.columns
        kept = set(cols)
        step_details = []
        all_itemsets = []

        for phase in range(1, len(cols) + 1):
            phase_info = {"phase": phase, "kept": [], "pruned": []}
            combos = [
                c for c in itertools.combinations(kept, phase)
                if len({cc.split("_P")[0] for cc in c}) == len(c)
            ]

            newly_kept = []
            for combo in combos:
                sig = calculate_significance(combo, df, weights, min_support=min_support)
                if sig >= min_support:
                    newly_kept.append(combo)
                    phase_info["kept"].append({"combination": combo, "significance": sig})
                else:
                    phase_info["pruned"].append({"combination": combo, "significance": sig})

            # Update
            kept = {item for combo in newly_kept for item in combo}
            all_itemsets.extend(newly_kept)
            step_details.append(phase_info)

            logger.debug(f"Phase {phase}: Kept {len(newly_kept)} combos, Pruned {len(phase_info['pruned'])} combos.")

            if not newly_kept:
                break

        return all_itemsets, step_details
    except Exception as e:
        logger.error(f"Error in determine_farm_rules: {e}")
        return [], []

def generate_rules(itemsets, df, weights, min_certainty=0.5):
    """
    From itemsets => FARM rules if certainty >= min_certainty.
    """
    try:
        def calc_certainty(antecedent, consequent):
            union_c = tuple(set(antecedent) | {consequent})
            sig_union = calculate_significance(union_c, df, weights)
            sig_antecedent = calculate_significance(antecedent, df, weights)
            return (sig_union / sig_antecedent) if sig_antecedent else 0

        rules = []
        for itemset in itemsets:
            if len(itemset) < 2:  # need at least 2 items
                continue
            for c in itemset:
                a = tuple(set(itemset) - {c})
                cf = calc_certainty(a, c)
                if cf >= min_certainty:
                    rules.append({
                        "antecedent": a,
                        "consequent": c,
                        "certainty": cf
                    })
        logger.debug(f"Generated {len(rules)} FARM rules.")
        print(rules)
        return rules
    except Exception as e:
        logger.error(f"Error in generate_rules: {e}")
        return []

def convert_farm_rules_html_to_english(farm_rules_html):
    """
    Converts FARM rules from HTML table format to understandable English sentences.

    Args:
        farm_rules_html (str): HTML string containing the FARM rules table.

    Returns:
        List[str]: A list of English sentences representing each rule.
    """
    try:
        soup = BeautifulSoup(farm_rules_html, 'html.parser')
        rules = []

        # Find the table with class 'farm-rules-table'
        table = soup.find('table', {'class': 'farm-rules-table'})
        if not table:
            raise ValueError("No table found with class 'farm-rules-table'.")

        # Extract table headers
        headers = [th.get_text(strip=True).lower() for th in table.find('thead').find_all('th')]
        required_headers = {'antecedent', 'consequent', 'certainty'}

        if not required_headers.issubset(set(headers)):
            raise ValueError(f"Table headers missing required columns: {required_headers}")

        # Map headers to their index positions
        header_indices = {header: index for index, header in enumerate(headers)}

        # Iterate over each row in the table body
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < len(required_headers):
                continue  # Skip incomplete rows

            antecedent = cells[header_indices['antecedent']].get_text(strip=True)
            consequent = cells[header_indices['consequent']].get_text(strip=True)
            certainty = cells[header_indices['certainty']].get_text(strip=True)

            # Convert certainty to a percentage for better readability
            try:
                certainty_value = float(certainty)
                certainty_percentage = f"{certainty_value * 100:.1f}%"
            except ValueError:
                certainty_percentage = certainty  # Keep as string if conversion fails

            # Construct English sentence
            sentence = (
                f"If {antecedent}, then {consequent} with a certainty of {certainty_percentage}."
            )
            rules.append(sentence)

        logger.debug(f"Converted HTML FARM rules to English sentences. Total rules: {len(rules)}")
        return rules
    except Exception as e:
        logger.error(f"Error in convert_farm_rules_html_to_english: {e}")
        return []

# ------------------------------------------------------
# Flask Routes
# ------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    """
    Renders the main template (e.g., base.html).
    All functionality is handled via AJAX in that template.
    """
    return render_template('base.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Fetch available AI models via ollama.list() and return model names as JSON.
    """
    try:
        logger.debug("Entered get_models route")
        model_data = str(ollama.list())
        pattern = r"model='(.*?)'"
        models = re.findall(pattern, model_data)
        models = [name for name in models if name.strip()]  # Clean up model names
        logger.debug(f"Available models: {models}")
        return jsonify({"success": True, "models": models})
    except Exception as e:
        logger.error(f"Error in get_models: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_model', methods=['POST'])
def set_model():
    """
    Set the selected Ollama model and store it in the session.
    """
    try:
        selected_model = request.form.get('model')
        if not selected_model:
            logger.warning("No model selected in set_model route.")
            return jsonify({"success": False, "error": "No model selected."}), 400

        # Store the selected model in the session
        session['selected_model'] = selected_model
        logger.debug(f"Selected Ollama model set to: {selected_model}")

        return jsonify({"success": True, "message": f"Model '{selected_model}' set successfully."})
    except Exception as e:
        logger.error(f"Error in set_model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    """
    Receive min_support and min_certainty from the fixed left menu,
    store them in session for later use in FARM logic.
    """
    try:
        min_support = request.form.get('min_support', type=float)
        min_certainty = request.form.get('min_certainty', type=float)

        if min_support is None or min_certainty is None:
            logger.warning("Thresholds not properly set in set_thresholds route.")
            return jsonify({"success": False, "error": "Invalid thresholds provided."}), 400

        # Store thresholds in session
        session['min_support'] = min_support
        session['min_certainty'] = min_certainty

        logger.debug(f"Updated thresholds => min_support: {min_support}, min_certainty: {min_certainty}")

        return jsonify({
            "success": True,
            "min_support": min_support,
            "min_certainty": min_certainty
        })
    except Exception as e:
        logger.error(f"Error in set_thresholds: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ajax_upload', methods=['POST'])
def ajax_upload():
    """
    Upload file, parse columns, build auto-ranges, create distribution charts.
    """
    try:
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            logger.warning("Invalid file format uploaded.")
            return jsonify({"success": False, "error": "Invalid file format. Please upload CSV or Excel."}), 400

        # Save file to the upload folder
        filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        filename_unique = f"{unique_id}_{filename}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename_unique)
        file.save(path)
        logger.debug(f"File uploaded and saved as: {filename_unique}")

        # Store the file path in the session for subsequent requests
        session['uploaded_file_path'] = path

        # Parse columns and generate distribution charts
        columns = get_columns(path)
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        charts = generate_distribution_charts(df, folder=app.config['PLOTS_FOLDER'])

        # Build auto ranges
        auto_ranges = {}
        for col in columns['numerical']:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            auto_ranges[col] = {"min": round(col_min, 3), "max": round(col_max, 3)}

        logger.debug("File processing completed successfully.")

        return jsonify({
            "success": True,
            "filepath": path,
            "columns": columns,
            "auto_ranges": auto_ranges,
            "chart_paths": charts,
        })
    except Exception as e:
        logger.error(f"Error in ajax_upload: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ajax_process', methods=['POST'])
def ajax_process():
    """
    After partition config is set, apply fuzzy logic, do FARM, return partial HTML + membership plots.
    Uses min_support and min_certainty from session if set, else defaults.
    """
    try:
        path = request.form.get('filepath')
        if not path or not os.path.exists(path):
            logger.warning("No valid file path provided in ajax_process route.")
            return jsonify({"success": False, "error": "No valid file path provided."}), 400

        selected_cols = request.form.getlist('columns')
        if not selected_cols:
            logger.warning("No columns selected for processing in ajax_process route.")
            return jsonify({"success": False, "error": "No columns selected for processing."}), 400

        # Retrieve thresholds from session; fallback to defaults if not set
        min_support = session.get('min_support', 0.2)     # Using user-set or default
        min_certainty = session.get('min_certainty', 0.5) # Using user-set or default
        logger.debug(f"Using thresholds => min_support: {min_support}, min_certainty: {min_certainty}")

        partitions_map, ranges_map, weights_map = {}, {}, {}

        # Parse partition config from form
        for col in selected_cols:
            part_count = int(request.form.get(f"partitions_{col}", 3))
            range_type = request.form.get(f"range_type_{col}", 'auto')
            user_min = request.form.get(f"min_{col}")
            user_max = request.form.get(f"max_{col}")

            if range_type == 'manual' and user_min and user_max:
                col_min = float(user_min)
                col_max = float(user_max)
            else:
                col_min = col_max = None

            partitions_map[col] = part_count
            ranges_map[col] = (col_min, col_max)

            # Weights
            weights_map[col] = {}
            for i in range(1, part_count + 1):
                w_key = f"weight_{col}_P{i}"
                w_val = request.form.get(w_key)
                try:
                    weights_map[col][f"{col}_P{i}"] = float(w_val) if w_val else 1.0
                except ValueError:
                    weights_map[col][f"{col}_P{i}"] = 1.0

        # Fuzzy logic
        fuzzy_df, membership_plots = apply_fuzzy_logic(path, partitions_map, ranges_map, weights_map)

        # FARM itemset analysis
        itemsets, step_details = determine_farm_rules(fuzzy_df, weights_map, min_support=min_support)
        rules = generate_rules(itemsets, fuzzy_df, weights_map, min_certainty=min_certainty)

        # Convert rules to HTML table
        rules_html = "<table class='farm-rules-table'><thead><tr><th>Antecedent</th><th>Consequent</th><th>Certainty</th></tr></thead><tbody>"
        for rule in rules:
            antecedent = ', '.join(rule['antecedent'])
            consequent = rule['consequent']
            certainty = f"{rule['certainty']:.4f}"
            rules_html += f"<tr><td>{antecedent}</td><td>{consequent}</td><td>{certainty}</td></tr>"
        rules_html += "</tbody></table>"

        # Save rules_html to a temporary file and store the filename in session
        rules_id = uuid.uuid4().hex
        rules_filename = f"{rules_id}_farm_rules.html"
        rules_filepath = os.path.join(app.config['UPLOAD_FOLDER'], rules_filename)
        with open(rules_filepath, 'w', encoding='utf-8') as f:
            f.write(rules_html)
        session['farm_rules_file'] = rules_filename
        logger.debug(f"FARM rules saved to file: {rules_filename}")

        # Build partial HTML for processed dataset
        results_html = fuzzy_df.to_html(classes='table table-bordered', index=False)

        # Build HTML for step-by-step itemsets
        steps_html = ""
        for phase in step_details:
            p_num = phase["phase"]
            steps_html += f"<div><h4>Phase {p_num}: {p_num}-Itemsets</h4>"
            steps_html += "<table class='combinations-table'>"
            steps_html += "<thead><tr><th>Combination</th><th>Significance</th><th>Status</th></tr></thead><tbody>"
            for k in phase["kept"]:
                combination = ', '.join(k['combination'])
                significance = f"{k['significance']:.4f}"
                steps_html += (
                    f"<tr><td>{combination}</td>"
                    f"<td>{significance}</td>"
                    f"<td style='color:green;'>Kept</td></tr>"
                )
            for p in phase["pruned"]:
                combination = ', '.join(p['combination'])
                significance = f"{p['significance']:.4f}"
                steps_html += (
                    f"<tr><td>{combination}</td>"
                    f"<td>{significance}</td>"
                    f"<td style='color:red;'>Pruned</td></tr>"
                )
            steps_html += "</tbody></table></div>"

        # Store references in session
        session['selected_columns'] = selected_cols
        session['partitions_map'] = partitions_map
        session['ranges_map'] = ranges_map
        session['weights_map'] = weights_map

        logger.debug("AJAX processing completed successfully.")

        # Return JSON for AJAX
        return jsonify({
            "success": True,
            "results_html": results_html,
            "rules_html": rules_html,  # For frontend display
            "steps_html": steps_html,
            "plots": membership_plots
        })
    except Exception as e:
        logger.error(f"Error in ajax_process: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/explore_farm_rules', methods=['POST'])
def explore_farm_rules():
    """
    Explore FARM rules by varying support and certainty thresholds.
    Generates a heatmap of the number of rules generated.
    """
    try:
        logger.debug("Starting explore_farm_rules...")

        step_support = float(request.form.get('step_support', 0.1))
        step_certainty = float(request.form.get('step_certainty', 0.1))
        logger.debug(f"Received step sizes -> step_support: {step_support}, step_certainty: {step_certainty}")

        path = session.get('uploaded_file_path')
        if not path or not os.path.exists(path):
            logger.error("Invalid or missing file path in explore_farm_rules route.")
            return jsonify({"success": False, "error": "No valid file path provided."}), 400

        selected_cols = session.get('selected_columns', [])
        partitions_map = session.get('partitions_map', {})
        ranges_map = session.get('ranges_map', {})
        weights_map = session.get('weights_map', {})
        logger.debug(f"Selected columns: {selected_cols}")
        logger.debug(f"Partitions map: {partitions_map}")
        logger.debug(f"Ranges map: {ranges_map}")
        logger.debug(f"Weights map: {weights_map}")

        # Apply fuzzy logic
        fuzzy_df, _ = apply_fuzzy_logic(path, partitions_map, ranges_map, weights_map)

        supports = np.arange(session.get('min_support', 0.2), 1.01, step_support)
        certainties = np.arange(session.get('min_certainty', 0.5), 1.01, step_certainty)
        logger.debug(f"Supports: {supports}")
        logger.debug(f"Certainties: {certainties}")

        heatmap_data = np.zeros((len(supports), len(certainties)))
        rule_details = []

        for i, support in enumerate(supports):
            for j, certainty in enumerate(certainties):
                itemsets, _ = determine_farm_rules(fuzzy_df, weights_map, min_support=support)
                rules = generate_rules(itemsets, fuzzy_df, weights_map, min_certainty=certainty)

                heatmap_data[i, j] = len(rules)
                rule_details.append({
                    "support": round(support, 2),
                    "certainty": round(certainty, 2),
                    "rule_count": len(rules),
                    "rules": rules,
                })

        # Plot heatmap with annotations
        heatmap_filename = "farm_rule_heatmap.png"
        heatmap_path = os.path.join(app.config['PLOTS_FOLDER'], heatmap_filename)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(heatmap_data, cmap="Blues", interpolation="nearest", origin="lower")
        plt.colorbar(im, label="Number of FARM Rules")
        plt.xticks(range(len(certainties)), [f"{c:.2f}" for c in certainties], rotation=45)
        plt.yticks(range(len(supports)), [f"{s:.2f}" for s in supports])
        plt.xlabel("Certainty Threshold")
        plt.ylabel("Support Threshold")
        plt.title("FARM Rules Heatmap")

        # Add annotations
        for i in range(len(supports)):
            for j in range(len(certainties)):
                plt.text(j, i, int(heatmap_data[i, j]),
                         ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

        logger.debug(f"Heatmap saved as {heatmap_filename}")

        # Build response
        response_data = {
            "success": True,
            "summary": f"Explored {len(supports) * len(certainties)} combinations.",
            "details": "<h3>Detailed FARM Rules</h3>",  # Can be expanded as needed
            "plot_filename": heatmap_filename,
        }
        logger.debug(f"Response data: {response_data}")

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in explore_farm_rules: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
def convert_farm_rules_html_to_structured_data(farm_rules_html):
    """
    Converts FARM rules from HTML table format to a structured data format.

    Args:
        farm_rules_html (str): HTML string containing the FARM rules table.

    Returns:
        List[dict]: A list of dictionaries, each representing a FARM rule with
                    keys: 'antecedent', 'consequent', and 'certainty'.
    """
    try:
        soup = BeautifulSoup(farm_rules_html, 'html.parser')
        structured_rules = []

        # Find the table with class 'farm-rules-table'
        table = soup.find('table', {'class': 'farm-rules-table'})
        if not table:
            raise ValueError("No table found with class 'farm-rules-table'.")

        # Extract table headers
        headers = [th.get_text(strip=True).lower() for th in table.find('thead').find_all('th')]
        required_headers = {'antecedent', 'consequent', 'certainty'}

        if not required_headers.issubset(set(headers)):
            raise ValueError(f"Table headers missing required columns: {required_headers}")

        # Map headers to their index positions
        header_indices = {header: index for index, header in enumerate(headers)}

        # Iterate over each row in the table body
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < len(required_headers):
                continue  # Skip incomplete rows

            antecedent = cells[header_indices['antecedent']].get_text(strip=True)
            consequent = cells[header_indices['consequent']].get_text(strip=True)
            certainty = cells[header_indices['certainty']].get_text(strip=True)

            # Convert antecedent into a list (split by commas)
            antecedent_list = [item.strip() for item in antecedent.split(',') if item.strip()]

            # Convert certainty to a float
            try:
                certainty_value = float(certainty)
            except ValueError:
                logger.warning(f"Invalid certainty value: {certainty}. Skipping rule.")
                continue

            # Add the rule as a dictionary
            structured_rules.append({
                'antecedent': antecedent_list,
                'consequent': consequent,
                'certainty': certainty_value
            })

        logger.debug(f"Converted HTML FARM rules to structured data. Total rules: {len(structured_rules)}")
        return structured_rules
    except Exception as e:
        logger.error(f"Error in convert_farm_rules_html_to_structured_data: {e}")
        return []

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """
    Generate a summary of FARM rules using the selected Ollama AI model.
    Rules are categorized by antecedent count, sorted by certainty,
    and limited to the top 10 rules per category.
    """
    try:
        # Retrieve FARM rules file from session
        farm_rules_filename = session.get('farm_rules_file')
        if not farm_rules_filename:
            logger.warning("No FARM rules available for summary generation.")
            return jsonify({"success": False, "error": "No FARM rules available. Please process data first."}), 400

        farm_rules_filepath = os.path.join(app.config['UPLOAD_FOLDER'], farm_rules_filename)
        if not os.path.exists(farm_rules_filepath):
            logger.error("FARM rules file does not exist.")
            return jsonify({"success": False, "error": "FARM rules file not found."}), 400

        # Read the FARM rules HTML from the file
        with open(farm_rules_filepath, 'r', encoding='utf-8') as f:
            farm_rules_html = f.read()

        # Convert HTML FARM rules to structured data
        rules = convert_farm_rules_html_to_structured_data(farm_rules_html)
        if not rules:
            logger.warning("No valid FARM rules found to summarize.")
            return jsonify({"success": False, "error": "No valid FARM rules found to summarize."}), 400

        # Categorize rules by antecedent count and sort by certainty
        categorized_rules = {}
        for rule in rules:
            num_antecedents = len(rule['antecedent'])
            certainty = rule['certainty']
            if num_antecedents not in categorized_rules:
                categorized_rules[num_antecedents] = []
            categorized_rules[num_antecedents].append((rule, certainty))

        # Sort each category by certainty in descending order and limit to top 10
        for key in categorized_rules:
            categorized_rules[key] = sorted(categorized_rules[key], key=lambda x: x[1], reverse=True)[:10]

        # Create formatted prompts for each category
        prompts = []
        for num_antecedents in sorted(categorized_rules.keys(), reverse=True):
            category_header = f"Rules with {num_antecedents} antecedent(s):"
            rules_text = "\n".join([
                f"{idx + 1}. IF {', '.join(rule['antecedent'])} THEN {rule['consequent']} "
                f"[Certainty: {certainty:.4f}]"
                for idx, (rule, certainty) in enumerate(categorized_rules[num_antecedents])
            ])
            prompts.append(f"{category_header}\n{rules_text}")

        full_prompt = "\n\n".join(prompts)

        logger.debug("Generated categorized and sorted FARM rules for summarization with top 10 rules per category.")

        # Retrieve the selected Ollama model from session
        selected_model = session.get('selected_model')
        if not selected_model:
            logger.warning("No Ollama model selected for summary generation.")
            return jsonify({"success": False, "error": "No Ollama model selected. Please set a model first."}), 400

        # Retrieve the modified prompt from session, if any
        modified_prompt = session.get('modified_prompt')

        # Define the improved prompt template with a placeholder
        default_prompt_template = (
            "We are conducting Fuzzy Associative Rule Mining (FARM) using the Apriori algorithm. "
            "The rules have been categorized based on the number of antecedents and arranged "
            "in descending order by the number of antecedents, then further sorted by certainty. "
            "Here are the top 10 rules for each category. Kindly analyze these rules and provide a concise summary. "
            "Ensure the summary captures the essential relationships, key patterns, and insights reflected by the rules. "
            "Highlight notable trends, significant partitions (_P1, _P2, etc.), and any recurring relationships. "
            "Write in a systematic fashion with clear line breaks for readability. EXPLAIN EACH AND EVERY RULE::::\n\n{rules}"
        )

        if modified_prompt:
            # Ensure the modified prompt contains the {rules} placeholder
            if "{rules}" not in modified_prompt:
                # If not, append the rules at the end
                prompt = f"{modified_prompt}\n\n{full_prompt}"
                logger.debug("Using modified prompt without placeholder.")
            else:
                # Replace the placeholder with actual rules
                prompt = modified_prompt.format(rules=full_prompt)
                logger.debug("Using modified prompt from session.")
        else:
            # Use the improved default prompt template
            prompt = default_prompt_template.format(rules=full_prompt)
            logger.debug("Using improved default prompt for summary generation.")

        logger.debug(f"Sending prompt to Ollama model '{selected_model}': {prompt[:100]}...")  # Truncated for brevity
        print(prompt)

        # Interact with the Ollama model
        response = ollama.chat(model=selected_model, messages=[{'role': 'user', 'content': prompt}])

        # Extract the summary from the AI response
        summary = response['message']['content']
        if not summary:
            logger.error("Received empty summary from Ollama model.")
            return jsonify({"success": False, "error": "Received empty summary from Ollama model."}), 500

        logger.debug("Summary generated successfully.")

        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        logger.error(f"Error in generate_summary: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/set_modified_prompt', methods=['POST'])
def set_modified_prompt():
    """
    Receive the modified prompt from the frontend and store it in the session.
    """
    modified_prompt = request.form.get('modified_prompt', '').strip()
    if not modified_prompt:
        logger.warning("Empty prompt received in set_modified_prompt route.")
        return jsonify({"success": False, "error": "Prompt cannot be empty."}), 400

    # Validate that the prompt contains the required placeholder
    if "{rules}" not in modified_prompt:
        logger.warning("Invalid prompt received. Missing '{rules}' placeholder.")
        return jsonify({"success": False, "error": "Prompt must include the '{rules}' placeholder."}), 400

    # Store the modified prompt in the session
    session['modified_prompt'] = modified_prompt
    logger.debug("Modified prompt set successfully.")
    return jsonify({"success": True, "message": "Prompt modified successfully."})

@app.route('/get_current_prompt', methods=['GET'])
def get_current_prompt():
    """
    Retrieve the current prompt (modified or default) to display in the Modify Prompt modal.
    """
    try:
        # Check for FARM rules availability
        farm_rules_filename = session.get('farm_rules_file')
        if not farm_rules_filename:
            logger.warning("No FARM rules available to generate prompt.")
            return jsonify({"success": False, "error": "No FARM rules available. Please process data first."}), 400

        # Check for the selected model in session
        selected_model = session.get('selected_model')
        if not selected_model:
            logger.warning("No Ollama model selected to generate prompt.")
            return jsonify({"success": False, "error": "No Ollama model selected. Please set a model first."}), 400

        # Retrieve either the modified prompt or use the default template
        modified_prompt = session.get('modified_prompt')
        if modified_prompt:
            prompt = modified_prompt
            logger.debug("Returning modified prompt from session.")
        else:
            default_prompt_template = (
                "We are conducting Fuzzy Associative Rule Mining (FARM) using the Apriori algorithm. "
                "The rules have been categorized based on the number of antecedents and arranged "
                "in descending order by the number of antecedents, then further sorted by certainty. "
                "Here are the top 10 rules for each category. Kindly analyze these rules and provide a concise summary. "
                "Ensure the summary captures the essential relationships, key patterns, and insights reflected by the rules. "
                "Highlight notable trends, significant partitions (_P1, _P2, etc.), and any recurring relationships. "
                "Write in a systematic fashion with clear line breaks for readability. EXPLAIN EACH AND EVERY RULE::::\n\n{rules}"
            )
            prompt = default_prompt_template
            logger.debug("Returning default prompt.")

        return jsonify({"success": True, "prompt": prompt})
    except Exception as e:
        logger.error(f"Error in get_current_prompt: {e}")
        return jsonify({"success": False, "error": "An error occurred while retrieving the current prompt."}), 500


@app.route('/reset_modified_prompt', methods=['POST'])
def reset_modified_prompt():
    """
    Reset the modified prompt by removing it from the session.
    """
    try:
        # Remove the modified prompt from the session
        if 'modified_prompt' in session:
            session.pop('modified_prompt')
            logger.debug("Modified prompt reset successfully.")
        else:
            logger.warning("No modified prompt found in session to reset.")

        return jsonify({"success": True, "message": "Prompt reset to default."})
    except Exception as e:
        logger.error(f"Error in reset_modified_prompt: {e}")
        return jsonify({"success": False, "error": "An error occurred while resetting the modified prompt."}), 500

# ------------------------------------------------------
# Serve Uploaded Files (if needed)
# ------------------------------------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files. Use with caution and ensure proper security measures.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

import subprocess  # Ensure subprocess is imported at the top

def is_ollama_running():
    """
    Checks if Ollama is running by attempting to execute 'ollama list'.
    Returns True if Ollama responds, False otherwise.
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Return output as string
            timeout=5  # Timeout after 5 seconds
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Ollama check failed: {e}")
        return False



# Route to check AI readiness and available models
@app.route('/check_ai_readiness', methods=['GET'])
def check_ai_readiness():
    if not is_ollama_running():
        return jsonify({
            "ollama_ready": False,
            "models": [],
            "error": "Ollama is not running or not found in PATH."
        })

    try:
        # Fetch available models from Ollama
        model_data = str(ollama.list())  # Assume this returns the list of Model objects

        # Regular expression to match the model name
        pattern = r"model='(.*?)'"  # Captures content between model=' and '

        # Use re.findall to extract all matches
        models = re.findall(pattern, model_data)
        models = [name.strip() for name in models if name.strip()]  # Strip whitespace and filter out empty strings

        logger.debug(f"Installed Ollama AI Models: {models}")
        return jsonify({
            "ollama_ready": True,
            "models": models
        })
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return jsonify({
            "ollama_ready": True,
            "models": [],
            "error": f"Error fetching Ollama models: {e}"
        })

# ------------------------------------------------------
# Entry Point
# ------------------------------------------------------
if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
