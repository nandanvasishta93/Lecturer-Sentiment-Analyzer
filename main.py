from flask import Flask, request, jsonify, render_template_string, send_file
import os
import json
import tempfile
from werkzeug.utils import secure_filename
import logging

# Import our sentiment analyzer
try:
    from lecturer_sentiment_analyzer import LecturerSentimentAnalyzer
    print("✅ Lecturer Sentiment Analyzer imported successfully")
except ImportError as e:
    print(f"❌ Error: lecturer_sentiment_analyzer.py not found or has errors: {e}")
    print("Please make sure the lecturer_sentiment_analyzer.py file is in the same directory.")
    exit(1)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an instance of our analyzer
try:
    analyzer = LecturerSentimentAnalyzer()
    logger.info("✅ Lecturer Sentiment Analyzer initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize analyzer: {e}")
    analyzer = None


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# HTML Template (embedded since we don't have a separate templates folder initially)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lecturer Sentiment Analysis Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        body { background-color: #f5f8fa; }
        .dashboard-card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: transform 0.3s;
        }
        .dashboard-card:hover { transform: translateY(-5px); }
        .header-section {
            background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .feedback-item {
            padding: 15px;
            border-left: 4px solid #4b6cb7;
            margin-bottom: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .alert-info { background-color: #e3f2fd; border-color: #90caf9; }
        .spinner-border { margin: 20px auto; display: block; }
        .file-info {
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header-section">
        <div class="container">
            <h1><i class="fas fa-microphone-alt me-2"></i>Lecturer Sentiment Analysis</h1>
            <p class="lead">AI-powered feedback to improve teaching effectiveness</p>
        </div>
    </div>

    <div class="container">
        <!-- Upload Section -->
        <div class="row mb-4">
            <div class="col-md-8 offset-md-2">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h4 class="card-title text-center">Upload Audio for Analysis</h4>
                        <p class="card-text text-center">Upload your lecture recording (WAV, MP3, OGG, M4A, FLAC) - Max 50MB</p>

                        <form id="uploadForm" enctype="multipart/form-data" class="text-center">
                            <div class="mb-3">
                                <input type="file" class="form-control" id="audioFile" accept="audio/*" required>
                                <div class="form-text">Supported formats: WAV, MP3, OGG, M4A, FLAC</div>
                            </div>
                            <button type="submit" class="btn btn-primary btn-lg">
                                <i class="fas fa-upload me-2"></i>Analyze Audio
                            </button>
                        </form>

                        <div id="uploadProgress" class="d-none mt-3">
                            <div class="alert alert-info text-center">
                                <div class="spinner-border text-primary mb-2" role="status">
                                    <span class="visually-hidden">Processing...</span>
                                </div>
                                <p class="mb-0">Processing your audio file... This may take a few minutes.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Status -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">System Status</h5>
                        <div class="row">
                            <div class="col-md-4">
                                <span id="analyzerStatus" class="badge bg-success">Analyzer Ready</span>
                                <small class="text-muted d-block">Speech Recognition</small>
                            </div>
                            <div class="col-md-4">
                                <span id="micStatus" class="badge bg-warning">Microphone Check</span>
                                <small class="text-muted d-block">Live Recording</small>
                            </div>
                            <div class="col-md-4">
                                <span class="badge bg-info">Web Interface</span>
                                <small class="text-muted d-block">Operational</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results-section" class="d-none">
            <!-- Summary Cards -->
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card dashboard-card text-center">
                        <div class="card-body">
                            <h6 class="text-muted">SENTIMENT</h6>
                            <div id="sentimentValue" class="metric-value text-primary">Neutral</div>
                            <small id="sentimentScore" class="text-muted">Score: 0.0</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card dashboard-card text-center">
                        <div class="card-body">
                            <h6 class="text-muted">SPEAKING PACE</h6>
                            <div id="paceValue" class="metric-value text-success">0</div>
                            <small class="text-muted">words/minute</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card dashboard-card text-center">
                        <div class="card-body">
                            <h6 class="text-muted">FILLER WORDS</h6>
                            <div id="fillerValue" class="metric-value text-warning">0%</div>
                            <small id="fillerCount" class="text-muted">0 occurrences</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card dashboard-card text-center">
                        <div class="card-body">
                            <h6 class="text-muted">VOCABULARY</h6>
                            <div id="vocabValue" class="metric-value text-info">0.0</div>
                            <small class="text-muted">richness score</small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Transcript and Feedback -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card dashboard-card">
                        <div class="card-body">
                            <h5 class="card-title">Lecture Transcript</h5>
                            <div class="p-3 bg-light rounded" style="height: 300px; overflow-y: auto;">
                                <p id="transcriptText" class="mb-0">Transcript will appear here...</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card dashboard-card">
                        <div class="card-body">
                            <h5 class="card-title">Improvement Suggestions</h5>
                            <div id="feedbackContainer" style="height: 300px; overflow-y: auto;">
                                <div class="text-muted">Feedback will appear here after analysis...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="row mb-4">
                <div class="col-12 text-center">
                    <button id="newAnalysis" class="btn btn-success">
                        <i class="fas fa-redo me-2"></i>Analyze Another File
                    </button>
                    <button id="downloadResults" class="btn btn-outline-primary ms-2">
                        <i class="fas fa-download me-2"></i>Download Results
                    </button>
                </div>
            </div>
        </div>

        <!-- Error Section -->
        <div id="error-section" class="d-none">
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error!</h4>
                <p id="errorMessage">An error occurred during analysis.</p>
            </div>
            <div class="text-center">
                <button id="tryAgain" class="btn btn-warning">
                    <i class="fas fa-redo me-2"></i>Try Again
                </button>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>

    <script>
        $(document).ready(function() {
            // Check system status on load
            checkSystemStatus();

            $('#uploadForm').on('submit', function(e) {
                e.preventDefault();

                const fileInput = document.getElementById('audioFile');
                const file = fileInput.files[0];

                if (!file) {
                    alert('Please select an audio file');
                    return;
                }

                // Validate file size (50MB)
                if (file.size > 50 * 1024 * 1024) {
                    alert('File size exceeds 50MB limit. Please choose a smaller file.');
                    return;
                }

                // Show progress
                $('#uploadProgress').removeClass('d-none');
                $('#results-section').addClass('d-none');
                $('#error-section').addClass('d-none');

                // Create FormData
                const formData = new FormData();
                formData.append('audio', file);

                // Upload and analyze
                $.ajax({
                    url: '/api/analyze',
                    type: 'POST',
                    data: formData,
                    contentType: false,
                    processData: false,
                    success: function(response) {
                        $('#uploadProgress').addClass('d-none');
                        displayResults(response);
                    },
                    error: function(xhr, status, error) {
                        $('#uploadProgress').addClass('d-none');
                        showError('Analysis failed: ' + (xhr.responseJSON?.error || error));
                    }
                });
            });

            $('#newAnalysis').on('click', function() {
                $('#results-section').addClass('d-none');
                $('#error-section').addClass('d-none');
                $('#audioFile').val('');
            });

            $('#tryAgain').on('click', function() {
                $('#error-section').addClass('d-none');
            });

            $('#downloadResults').on('click', function() {
                // In a real implementation, this would download a JSON report
                alert('Download feature would save analysis results as JSON file');
            });
        });

        function checkSystemStatus() {
            $.get('/api/health', function(response) {
                if (!response.analyzer_ready) {
                    $('#analyzerStatus').removeClass('bg-success').addClass('bg-danger').text('Analyzer Error');
                }
                // Note: Mic status would need additional endpoint to check
            }).fail(function() {
                $('#analyzerStatus').removeClass('bg-success').addClass('bg-danger').text('Service Unavailable');
            });
        }

        function displayResults(data) {
            // Update metrics
            if (data.sentiment) {
                $('#sentimentValue').text(data.sentiment.category || 'Unknown');
                $('#sentimentScore').text('Score: ' + (data.sentiment.polarity || 0).toFixed(2));
                
                // Color code sentiment
                const sentimentElem = $('#sentimentValue');
                sentimentElem.removeClass('text-primary text-success text-danger text-warning');
                switch(data.sentiment.category) {
                    case 'Positive': sentimentElem.addClass('text-success'); break;
                    case 'Negative': sentimentElem.addClass('text-danger'); break;
                    case 'Neutral': sentimentElem.addClass('text-primary'); break;
                    default: sentimentElem.addClass('text-warning');
                }
            }

            if (data.metrics) {
                $('#paceValue').text(Math.round(data.metrics.speaking_rate || 0));
                $('#fillerValue').text(Math.round((data.metrics.filler_ratio || 0) * 100) + '%');
                $('#fillerCount').text((data.metrics.filler_count || 0) + ' occurrences');
                $('#vocabValue').text((data.metrics.vocabulary_richness || 0).toFixed(2));
            }

            // Update transcript
            $('#transcriptText').text(data.transcript || 'No transcript available');

            // Update feedback
            const feedbackContainer = $('#feedbackContainer');
            feedbackContainer.empty();

            if (data.feedback && data.feedback.length > 0) {
                data.feedback.forEach(function(item) {
                    feedbackContainer.append('<div class="feedback-item">' + item + '</div>');
                });
            } else {
                feedbackContainer.append('<div class="text-muted">No specific feedback available.</div>');
            }

            // Show results
            $('#results-section').removeClass('d-none');
        }

        function showError(message) {
            $('#errorMessage').text(message);
            $('#error-section').removeClass('d-none');
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """API endpoint to analyze an uploaded audio file"""
    if not analyzer:
        return jsonify({'error': 'Analyzer not initialized properly'}), 500

    # Check if the post request has the file part
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']

    # If user does not select file, browser submits an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save the uploaded file
            file.save(file_path)
            logger.info(f"File saved: {file_path}")

            # Run analysis on the saved file
            results = analyzer.run_analysis_from_file(file_path)

            # Clean up - delete the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file deleted: {file_path}")

            if results:
                logger.info("Analysis completed successfully")
                return jsonify(results)
            else:
                return jsonify({'error': 'Analysis failed - no results generated'}), 500

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            # Clean up file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Please upload WAV, MP3, OGG, M4A, or FLAC files.'}), 400


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'analyzer_ready': analyzer is not None,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'mic_available': analyzer.mic_available if analyzer else False
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error occurred'}), 500


if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("=" * 60)
    print("🎤 LECTURER SENTIMENT ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"🚀 Server starting on: http://localhost:5000")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"🔧 Analyzer status: {'✅ Ready' if analyzer else '❌ Not initialized'}")
    if analyzer:
        print(f"🎤 Microphone: {'✅ Available' if analyzer.mic_available else '❌ Not available'}")
    print("=" * 60)
    print("📋 Instructions:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Upload an audio file of your lecture")
    print("   3. View analysis results and feedback")
    print("=" * 60)

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)