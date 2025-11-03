import os
import time
import json
import numpy as np
import pandas as pd
from textblob import TextBlob
import speech_recognition as sr
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources if not already present"""
    resources = [
        'punkt',           # Tokenizer
        'stopwords',       # Stopwords corpus
        'averaged_perceptron_tagger',  # POS tagger
    ]
    
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✅ NLTK {resource} downloaded/available")
        except Exception as e:
            print(f"⚠️  Could not download {resource}: {e}")
            # Continue with other resources


# Initialize NLTK resources
download_nltk_resources()


class LecturerSentimentAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # Initialize microphone only if available
        try:
            self.microphone = sr.Microphone()
            self.mic_available = True
            print("✅ Microphone available for live recording")
        except OSError:
            print("⚠️  No microphone detected. Live recording disabled.")
            self.microphone = None
            self.mic_available = False
            
        self.filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'well', 'okay']

        # Initialize stopwords safely
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"⚠️  Could not load stopwords: {e}")
            # Fallback stopwords
            self.stop_words = set([
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            ])

        self.results = {
            'transcript': '',
            'sentiment': {},
            'metrics': {},
            'feedback': []
        }

        # Live recording variables
        self.is_recording = False
        self.live_transcript = ""
        self.start_time = None

    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        if not self.mic_available:
            print("❌ Microphone not available for calibration")
            return False
            
        print("Calibrating microphone for ambient noise...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Microphone calibrated successfully.")
                return True
        except Exception as e:
            print(f"❌ Could not calibrate microphone: {e}")
            return False

    def transcribe_audio(self, audio_file):
        """Convert audio file to text using Speech Recognition"""
        print(f"Transcribing audio file: {audio_file}")

        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return ""

        try:
            # Check if file is a valid audio file
            if not audio_file.lower().endswith(('.wav', '.flac', '.aiff', '.mp3', '.m4a', '.ogg')):
                print("Warning: File format may not be optimal. Supported formats: WAV, FLAC, AIFF")

            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Reading audio data...")
                audio_data = self.recognizer.record(source)

                print("Converting speech to text...")
                transcript = self.recognizer.recognize_google(audio_data)
                self.results['transcript'] = transcript
                print("✅ Transcription complete.")
                return transcript

        except sr.UnknownValueError:
            error_msg = "Could not understand audio clearly. Please try with a clearer recording."
            print(f"Warning: {error_msg}")
            self.results['transcript'] = error_msg
            return self.results['transcript']
        except sr.RequestError as e:
            error_msg = f"Speech recognition service error: {e}"
            print(f"Error: {error_msg}")
            self.results['transcript'] = error_msg
            return self.results['transcript']
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            print(f"Error: {error_msg}")
            self.results['transcript'] = error_msg
            return self.results['transcript']

    def start_live_recording(self, update_interval=10):
        """Start live recording and analysis"""
        if not self.mic_available:
            print("❌ Live recording not available - no microphone detected")
            return

        self.is_recording = True
        self.start_time = time.time()
        self.live_transcript = ""

        print("Starting live recording... Press Ctrl+C to stop.")
        if not self.calibrate_microphone():
            return

        try:
            while self.is_recording:
                try:
                    # Record audio for the specified interval
                    with self.microphone as source:
                        audio_data = self.recognizer.listen(source, timeout=1, phrase_time_limit=update_interval)

                    # Transcribe the audio chunk
                    try:
                        chunk_text = self.recognizer.recognize_google(audio_data)
                        self.live_transcript += " " + chunk_text
                        print(f"Recognized: {chunk_text}")

                        # Update analysis
                        self.analyze_live_sentiment()

                    except sr.UnknownValueError:
                        print("Could not understand audio chunk")
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")

                except sr.WaitTimeoutError:
                    # No speech detected in timeout period
                    pass
                except KeyboardInterrupt:
                    print("\nStopping recording...")
                    break

        except Exception as e:
            print(f"Error during live recording: {e}")
        finally:
            self.stop_live_recording()

    def stop_live_recording(self):
        """Stop live recording and perform final analysis"""
        self.is_recording = False
        if self.live_transcript.strip():
            self.results['transcript'] = self.live_transcript.strip()
            self.analyze_sentiment()
            self.generate_feedback()
            print("\n✅ Final analysis completed.")
        else:
            print("No speech was detected during recording.")

    def analyze_live_sentiment(self):
        """Analyze sentiment during live recording"""
        if not self.live_transcript.strip():
            return

        # Perform quick analysis for live updates
        blob = TextBlob(self.live_transcript)
        sentiment_polarity = blob.sentiment.polarity

        if sentiment_polarity > 0.1:
            sentiment_category = "Positive"
        elif sentiment_polarity < -0.1:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"

        # Calculate live metrics
        words = self.safe_tokenize(self.live_transcript)
        word_count = len(words)
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        speaking_rate = (word_count / elapsed_time) * 60 if elapsed_time > 0 else 0

        # Count filler words
        filler_count = sum(1 for word in words if word.lower().strip('.,!?') in self.filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0

        # Update live results
        self.results.update({
            'live_sentiment': {
                'category': sentiment_category,
                'polarity': float(sentiment_polarity)
            },
            'live_metrics': {
                'word_count': word_count,
                'speaking_rate': float(speaking_rate),
                'filler_count': filler_count,
                'filler_ratio': float(filler_ratio),
                'session_time': float(elapsed_time)
            }
        })

        # Print live update
        print(f"\n--- LIVE UPDATE ---")
        print(f"Sentiment: {sentiment_category} ({sentiment_polarity:.2f})")
        print(f"Speaking Rate: {speaking_rate:.1f} words/minute")
        print(f"Filler Words: {filler_ratio * 100:.1f}% ({filler_count} occurrences)")
        print(f"Session Time: {elapsed_time / 60:.1f} minutes")
        print(f"Current transcript length: {word_count} words")

    def analyze_sentiment(self, text=None):
        """Analyze sentiment of the transcribed text"""
        if text is None:
            text = self.results['transcript']

        if not text or len(text.strip()) < 5:
            print("Warning: No sufficient text to analyze")
            self.results['sentiment'] = {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'category': 'Neutral'
            }
            self.calculate_basic_metrics(text or "")
            return self.results['sentiment']

        print("Analyzing sentiment...")

        try:
            # Sentiment analysis using TextBlob
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity

            # Categorize sentiment
            if sentiment_polarity > 0.1:
                sentiment_category = "Positive"
            elif sentiment_polarity < -0.1:
                sentiment_category = "Negative"
            else:
                sentiment_category = "Neutral"

            self.results['sentiment'] = {
                'polarity': float(sentiment_polarity),
                'subjectivity': float(sentiment_subjectivity),
                'category': sentiment_category
            }

            # Calculate additional metrics
            self.calculate_metrics(text)

            print("✅ Sentiment analysis complete.")
            return self.results['sentiment']

        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            self.results['sentiment'] = {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'category': 'Neutral'
            }
            self.calculate_basic_metrics(text)
            return self.results['sentiment']

    def calculate_metrics(self, text):
        """Calculate various speaking metrics from the transcript"""
        try:
            words = self.safe_tokenize(text)
            word_count = len(words)

            # Better duration estimation
            if hasattr(self, 'start_time') and self.start_time:
                actual_duration = time.time() - self.start_time
                estimated_duration = max(actual_duration, word_count / 2.5)
            else:
                # Estimate based on typical speaking rate (150 wpm)
                estimated_duration = max(60, word_count / 2.5)

            speaking_rate = (word_count / estimated_duration) * 60 if estimated_duration > 0 else 0

            # Filler word analysis
            filler_count = sum(1 for word in words if word.lower().strip('.,!?') in self.filler_words)
            filler_ratio = filler_count / word_count if word_count > 0 else 0

            # Sentence analysis
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentence_count = max(1, len(sentences))
            avg_sentence_length = word_count / sentence_count

            # Vocabulary analysis
            content_words = [w.lower() for w in words if
                             w.lower() not in self.stop_words and w.isalpha() and len(w) > 2]
            content_word_count = len(content_words)
            unique_words = len(set(content_words))
            vocabulary_richness = unique_words / content_word_count if content_word_count > 0 else 0

            # Store metrics
            self.results['metrics'] = {
                'word_count': word_count,
                'speaking_rate': float(speaking_rate),
                'filler_count': filler_count,
                'filler_ratio': float(filler_ratio),
                'avg_sentence_length': float(avg_sentence_length),
                'vocabulary_richness': float(vocabulary_richness),
                'duration_seconds': float(estimated_duration),
                'sentence_count': sentence_count,
                'unique_words': unique_words,
                'content_words': content_word_count
            }

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            self.calculate_basic_metrics(text)

    def calculate_basic_metrics(self, text):
        """Calculate basic metrics when full analysis fails"""
        try:
            words = text.split() if text else []
            word_count = len(words)

            self.results['metrics'] = {
                'word_count': word_count,
                'speaking_rate': 120.0,
                'filler_count': 0,
                'filler_ratio': 0.0,
                'avg_sentence_length': 15.0,
                'vocabulary_richness': 0.5,
                'duration_seconds': 60.0,
                'sentence_count': max(1, text.count('.')) if text else 1,
                'unique_words': len(set(words)) if words else 0,
                'content_words': word_count
            }
        except Exception as e:
            print(f"Error in basic metrics calculation: {e}")
            self.results['metrics'] = {
                'word_count': 0,
                'speaking_rate': 120.0,
                'filler_count': 0,
                'filler_ratio': 0.0,
                'avg_sentence_length': 15.0,
                'vocabulary_richness': 0.5,
                'duration_seconds': 60.0,
                'sentence_count': 1,
                'unique_words': 0,
                'content_words': 0
            }

    def safe_tokenize(self, text):
        """Safely tokenize text with fallback"""
        try:
            return word_tokenize(text.lower())
        except Exception as e:
            print(f"NLTK tokenization failed: {e}, using simple split")
            return text.lower().split()

    def generate_feedback(self):
        """Generate comprehensive feedback based on analysis"""
        print("Generating feedback...")
        feedback = []

        if not self.results.get('metrics'):
            feedback.append("Analysis completed, but detailed metrics are not available.")
            self.results['feedback'] = feedback
            return feedback

        try:
            sentiment = self.results.get('sentiment', {})
            metrics = self.results.get('metrics', {})

            # Sentiment feedback
            sentiment_category = sentiment.get('category', 'Neutral')
            sentiment_polarity = sentiment.get('polarity', 0)

            if sentiment_category == "Negative":
                feedback.append(
                    "🔴 Your tone appears somewhat negative. Consider using more positive and encouraging language.")
            elif sentiment_category == "Neutral" and sentiment_polarity < 0.05:
                feedback.append(
                    "🟡 Your tone is quite neutral. Adding more enthusiasm could increase engagement.")
            else:
                feedback.append(
                    "🟢 Your positive tone creates an encouraging learning environment.")

            # Speaking rate feedback
            speaking_rate = metrics.get('speaking_rate', 120)
            if speaking_rate > 180:
                feedback.append(f"⚡ Speaking rate: {speaking_rate:.0f} wpm - Too fast! Slow down for clarity.")
            elif speaking_rate < 90:
                feedback.append(
                    f"🐌 Speaking rate: {speaking_rate:.0f} wpm - Too slow. Increase pace to maintain attention.")
            else:
                feedback.append(f"✅ Speaking rate: {speaking_rate:.0f} wpm - Well-balanced pace.")

            # Filler word feedback
            filler_ratio = metrics.get('filler_ratio', 0)
            filler_count = metrics.get('filler_count', 0)

            if filler_ratio > 0.08:
                feedback.append(f"🚨 High filler word usage: {filler_ratio * 100:.1f}% ({filler_count} times)")
            elif filler_ratio > 0.03:
                feedback.append(f"⚠️ Moderate filler words: {filler_ratio * 100:.1f}% - Room for improvement")
            else:
                feedback.append("✨ Excellent! Very few filler words used.")

            # Vocabulary feedback
            vocabulary_richness = metrics.get('vocabulary_richness', 0.5)
            if vocabulary_richness < 0.4:
                feedback.append("📚 Consider using more varied vocabulary to maintain interest.")
            elif vocabulary_richness > 0.7:
                feedback.append("🎓 Rich vocabulary! Ensure it matches your audience level.")
            else:
                feedback.append("📖 Good vocabulary variety and accessibility.")

            # Session summary
            word_count = metrics.get('word_count', 0)
            duration = metrics.get('duration_seconds', 60)

            feedback.append(f"📊 Session Summary: {word_count} words in {duration / 60:.1f} minutes")

            # Motivational closing
            if len([f for f in feedback if "🟢" in f or "✅" in f or "✨" in f]) >= 2:
                feedback.append("🌟 Great job! You're demonstrating strong communication skills.")
            else:
                feedback.append("💪 Keep practicing! Small improvements make a big difference.")

        except Exception as e:
            print(f"Error generating feedback: {e}")
            feedback.append("Analysis completed. Continue working on clear communication.")

        self.results['feedback'] = feedback
        return feedback

    def save_results(self, filename="lecture_analysis.json"):
        """Save analysis results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def print_results(self):
        """Print formatted analysis results"""
        print("\n" + "=" * 50)
        print("LECTURE ANALYSIS RESULTS")
        print("=" * 50)

        # Transcript
        transcript = self.results.get('transcript', '')
        if transcript and not transcript.startswith('Could not'):
            print(f"\nTRANSCRIPT:\n{transcript[:200]}{'...' if len(transcript) > 200 else ''}")

        # Sentiment
        sentiment = self.results.get('sentiment', {})
        print(f"\nSENTIMENT ANALYSIS:")
        print(f"Category: {sentiment.get('category', 'N/A')}")
        print(f"Polarity: {sentiment.get('polarity', 0):.3f}")
        print(f"Subjectivity: {sentiment.get('subjectivity', 0):.3f}")

        # Metrics
        metrics = self.results.get('metrics', {})
        if metrics:
            print(f"\nSPEAKING METRICS:")
            print(f"Word Count: {metrics.get('word_count', 0)}")
            print(f"Speaking Rate: {metrics.get('speaking_rate', 0):.1f} wpm")
            print(f"Filler Words: {metrics.get('filler_count', 0)} ({metrics.get('filler_ratio', 0) * 100:.1f}%)")
            print(f"Avg Sentence Length: {metrics.get('avg_sentence_length', 0):.1f} words")
            print(f"Vocabulary Richness: {metrics.get('vocabulary_richness', 0):.3f}")

        # Feedback
        feedback = self.results.get('feedback', [])
        if feedback:
            print(f"\nFEEDBACK:")
            for i, fb in enumerate(feedback, 1):
                print(f"{i}. {fb}")

    def run_analysis_from_file(self, audio_file):
        """Run complete analysis on an audio file"""
        try:
            print(f"Starting analysis of: {audio_file}")

            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            # Get file info
            file_size = os.path.getsize(audio_file)
            print(f"File size: {file_size:,} bytes")

            # Transcribe and analyze
            transcript = self.transcribe_audio(audio_file)
            self.analyze_sentiment(transcript)
            self.generate_feedback()

            print("\n✅ Analysis completed successfully!")
            return self.results

        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            return self.create_error_results(str(e))

    def create_error_results(self, error_msg):
        """Create basic results structure for error cases"""
        return {
            'transcript': f"Analysis failed: {error_msg}",
            'sentiment': {'polarity': 0.0, 'subjectivity': 0.5, 'category': 'Neutral'},
            'metrics': {
                'word_count': 0, 'speaking_rate': 0, 'filler_count': 0,
                'filler_ratio': 0.0, 'avg_sentence_length': 0,
                'vocabulary_richness': 0, 'duration_seconds': 0,
                'sentence_count': 0, 'unique_words': 0, 'content_words': 0
            },
            'feedback': [f"❌ Analysis error: {error_msg}",
                         "Please check your audio file and try again."]
        }


# Demo and testing
if __name__ == "__main__":
    print("🎤 Lecturer Sentiment Analyzer - Enhanced Version")
    print("=" * 50)

    analyzer = LecturerSentimentAnalyzer()

    # Test with sample text
    print("\n📝 Testing with sample text...")
    sample_text = """
    Hello students, today we will learn about machine learning. 
    Um, it's a very interesting topic and, like, it has many applications in our daily lives.
    Machine learning is basically a subset of artificial intelligence that, well, 
    enables computers to learn without being explicitly programmed.
    """

    analyzer.results['transcript'] = sample_text
    analyzer.analyze_sentiment(sample_text)
    analyzer.generate_feedback()
    analyzer.print_results()

    print("\n" + "=" * 50)
    print("📋 Available Methods:")
    print("- analyzer.run_analysis_from_file('audio.wav')  # Analyze audio file")
    print("- analyzer.start_live_recording()              # Live recording mode")
    print("- analyzer.print_results()                     # Display results")
    print("- analyzer.save_results('results.json')        # Save to file")
    print("=" * 50)